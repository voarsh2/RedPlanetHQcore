import fs from "fs";
import path from "node:path";

import { logger } from "@trigger.dev/sdk/v3";
import {
  type CoreMessage,
  type LanguageModelV1,
  streamText,
  type ToolSet,
} from "ai";
import { createAmazonBedrock } from "@ai-sdk/amazon-bedrock";
import { fromNodeProviderChain } from "@aws-sdk/credential-providers";
import { createOllama, type OllamaProvider } from "ollama-ai-provider";
import { resolveProvider } from "~/lib/llm/providers";

import { type AgentMessageType, Message } from "./types";

interface State {
  inTag: boolean;
  messageEnded: boolean;
  message: string;
  lastSent: string;
}

export interface ExecutionState {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  agentFlow: any;
  userMessage: string;
  message: string;
}

export async function* processTag(
  state: State,
  totalMessage: string,
  chunk: string,
  startTag: string,
  endTag: string,
  states: { start: string; chunk: string; end: string },
  extraParams: Record<string, string> = {},
) {
  let comingFromStart = false;

  if (!state.messageEnded) {
    if (!state.inTag) {
      const startIndex = totalMessage.indexOf(startTag);
      if (startIndex !== -1) {
        state.inTag = true;
        // Send MESSAGE_START when we first enter the tag
        yield Message("", states.start as AgentMessageType, extraParams);
        const chunkToSend = totalMessage.slice(startIndex + startTag.length);
        state.message += chunkToSend;
        comingFromStart = true;
      }
    }

    if (state.inTag) {
      // Check if chunk contains end tag
      const hasEndTag = chunk.includes(endTag);
      const hasStartTag = chunk.includes(startTag);
      const hasClosingTag = chunk.includes("</");

      if (hasClosingTag && !hasStartTag && !hasEndTag) {
        // If chunk only has </ but not the full end tag, accumulate it
        state.message += chunk;
      } else if (hasEndTag || (!hasEndTag && !hasClosingTag)) {
        let currentMessage = comingFromStart
          ? state.message
          : state.message + chunk;

        const endIndex = currentMessage.indexOf(endTag);

        if (endIndex !== -1) {
          // For the final chunk before the end tag
          currentMessage = currentMessage.slice(0, endIndex).trim();
          const messageToSend = currentMessage.slice(
            currentMessage.indexOf(state.lastSent) + state.lastSent.length,
          );

          if (messageToSend) {
            yield Message(
              messageToSend,
              states.chunk as AgentMessageType,
              extraParams,
            );
          }
          // Send MESSAGE_END when we reach the end tag
          yield Message("", states.end as AgentMessageType, extraParams);

          state.message = currentMessage;
          state.messageEnded = true;
        } else {
          const diff = currentMessage.slice(
            currentMessage.indexOf(state.lastSent) + state.lastSent.length,
          );

          // For chunks in between start and end
          const messageToSend = comingFromStart ? state.message : diff;

          if (messageToSend) {
            state.lastSent = messageToSend;
            yield Message(
              messageToSend,
              states.chunk as AgentMessageType,
              extraParams,
            );
          }
        }

        state.message = currentMessage;
        state.lastSent = state.message;
      } else {
        state.message += chunk;
      }
    }
  }
}

export async function* generate(
  messages: CoreMessage[],
  isProgressUpdate: boolean = false,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onFinish?: (event: any) => void,
  tools?: ToolSet,
  system?: string,
  model?: string,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
): AsyncGenerator<
  | string
  | {
      type: string;
      toolName: string;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      args?: any;
      toolCallId?: string;
      message?: string;
    }
> {
  let ollamaUrl = process.env.OLLAMA_URL;
  model = model || process.env.MODEL;

  const temperatureOverride = process.env.MODEL_TEMPERATURE;
  let modelTemperature = temperatureOverride
    ? Number(temperatureOverride)
    : undefined;

  let ollama: OllamaProvider | undefined;
  if (ollamaUrl) {
    try {
      ollama = createOllama({
        baseURL: ollamaUrl,
      });
    } catch (error) {
      logger.warn("Unable to initialise Ollama provider", {
        error,
      });
    }
  }

  const bedrock = createAmazonBedrock({
    region: process.env.AWS_REGION || "us-east-1",
    credentialProvider: fromNodeProviderChain(),
  });

  const provider = resolveProvider(model as string, {
    bedrock,
    ollama,
  });

  if (provider?.defaultOptions?.temperature !== undefined && !temperatureOverride) {
    modelTemperature = Number(provider.defaultOptions.temperature);
  }

  const streamOptions =
    provider?.defaultOptions && Object.keys(provider.defaultOptions).length > 0
      ? { ...provider.defaultOptions }
      : {};

  if (modelTemperature !== undefined) {
    streamOptions.temperature = modelTemperature;
  }

  logger.info("starting stream");
  if (provider?.instance) {
    try {
      const { textStream, fullStream } = streamText({
        model: provider.instance as LanguageModelV1,
        messages,
        ...(streamOptions as Record<string, unknown>),
        maxSteps: 10,
        tools,
        ...(isProgressUpdate
          ? { toolChoice: { type: "tool", toolName: "core--progress_update" } }
          : {}),
        toolCallStreaming: true,
        onFinish,
        ...(system ? { system } : {}),
      });

      for await (const chunk of textStream) {
        yield chunk;
      }

      for await (const fullChunk of fullStream) {
        if (fullChunk.type === "tool-call") {
          yield {
            type: "tool-call",
            toolName: fullChunk.toolName,
            toolCallId: fullChunk.toolCallId,
            args: fullChunk.args,
          };
        }

        if (fullChunk.type === "error") {
          // Log the error to a file
          const errorLogsDir = path.join(__dirname, "../../../../logs/errors");

          // Ensure the directory exists
          try {
            if (!fs.existsSync(errorLogsDir)) {
              fs.mkdirSync(errorLogsDir, { recursive: true });
            }

            // Create a timestamped error log file
            const timestamp = new Date().toISOString().replace(/:/g, "-");
            const errorLogPath = path.join(
              errorLogsDir,
              `llm-error-${timestamp}.json`,
            );

            // Write the error to the file
            fs.writeFileSync(
              errorLogPath,
              JSON.stringify({
                timestamp: new Date().toISOString(),
                error: fullChunk.error,
              }),
            );

            logger.error(`LLM error logged to ${errorLogPath}`);
          } catch (err) {
            logger.error(`Failed to log LLM error: ${err}`);
          }
        }
      }
      return;
    } catch (e) {
      console.log(e);
      logger.error(e as string);
    }
  }

  throw new Error("No valid LLM configuration found");
}
