import {
  validateUIMessages,
  generateId,
  stepCountIs,
} from "ai";
import { z } from "zod";

import { createHybridActionApiRoute } from "~/services/routeBuilders/apiBuilder.server";
import {
  getConversationAndHistory,
  upsertConversationHistory,
} from "~/services/conversation.server";

import { makeModelCall } from "~/lib/model.server";
import { EpisodeType, UserTypeEnum } from "@core/types";
import { enqueueCreateConversationTitle } from "~/lib/queue-adapter.server";
import { addToQueue } from "~/lib/ingest.server";
import { buildAgentContext } from "~/services/agent/agent-context";
import { logger } from "~/services/logger.service";

const ChatRequestSchema = z.object({
  message: z
    .object({
      id: z.string().optional(),
      parts: z.array(z.any()),
      role: z.string(),
    })
    .optional(),
  messages: z
    .array(
      z.object({
        id: z.string().optional(),
        parts: z.array(z.any()),
        role: z.string(),
      }),
    )
    .optional(),
  id: z.string(),
  needsApproval: z.boolean().optional(),
  source: z.string().default("core"),
});

const summarizeParts = (parts: any[] | undefined, depth = 0): any[] => {
  if (!Array.isArray(parts) || depth > 2) {
    return [];
  }

  return parts.map((part: any) => {
    const nestedParts =
      part?.output?.parts ??
      part?.output?.content ??
      (Array.isArray(part?.output) ? part.output : undefined);

    return {
      type: part?.type,
      state: part?.state,
      textChars: typeof part?.text === "string" ? part.text.length : undefined,
      hasInput: Boolean(part?.input),
      hasOutput: Boolean(part?.output),
      outputShape: Array.isArray(part?.output)
        ? "array"
        : part?.output && typeof part.output === "object"
          ? Object.keys(part.output).slice(0, 8)
          : typeof part?.output,
      nested: summarizeParts(nestedParts, depth + 1),
    };
  });
};

const wrapChatResponseForDiagnostics = (
  response: Response,
  context: {
    conversationId: string;
    requestId: string;
    startedAt: number;
  },
) => {
  if (!response.body) {
    logger.warn(
      "Agent chat response has no body",
      {
        ...context,
        elapsedMs: Date.now() - context.startedAt,
        status: response.status,
      },
    );
    return response;
  }

  let chunkCount = 0;
  let byteCount = 0;
  let maxChunkBytes = 0;
  let lastProgressLogAt = Date.now();
  let firstChunkLogged = false;
  let closed = false;
  const reader = response.body.getReader();

  const stream = new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        const { done, value } = await reader.read();
        if (done) {
          closed = true;
          logger.info(
            "Agent chat UI stream closed",
            {
              ...context,
              elapsedMs: Date.now() - context.startedAt,
              chunkCount,
              byteCount,
              maxChunkBytes,
              status: response.status,
            },
          );
          controller.close();
          return;
        }

        const chunk = value;
        const chunkBytes = chunk.byteLength ?? chunk.length ?? 0;
        chunkCount += 1;
        byteCount += chunkBytes;
        maxChunkBytes = Math.max(maxChunkBytes, chunkBytes);

        if (!firstChunkLogged) {
          firstChunkLogged = true;
          logger.info(
            "Agent chat UI stream first chunk",
            {
              ...context,
              elapsedMs: Date.now() - context.startedAt,
              firstChunkBytes: chunkBytes,
              status: response.status,
            },
          );
        }

        const now = Date.now();
        if (
          chunkCount % 500 === 0 ||
          (chunkBytes > 500_000 && now - lastProgressLogAt > 5_000) ||
          now - lastProgressLogAt > 30_000
        ) {
          lastProgressLogAt = now;
          logger.info(
            "Agent chat UI stream progress",
            {
              ...context,
              elapsedMs: now - context.startedAt,
              chunkCount,
              byteCount,
              chunkBytes,
              maxChunkBytes,
              status: response.status,
            },
          );
        }

        controller.enqueue(chunk);
      } catch (error) {
        logger.error(
          "Agent chat UI stream failed while reading",
          {
            ...context,
            elapsedMs: Date.now() - context.startedAt,
            chunkCount,
            byteCount,
            maxChunkBytes,
            status: response.status,
            error,
          },
        );
        controller.error(error);
      }
    },
    async cancel(reason) {
      if (!closed) {
        logger.warn(
          "Agent chat UI stream canceled by client",
          {
            ...context,
            elapsedMs: Date.now() - context.startedAt,
            chunkCount,
            byteCount,
            maxChunkBytes,
            status: response.status,
            reason,
          },
        );
      }
      await reader.cancel(reason);
    },
  });

  return new Response(stream, {
    status: response.status,
    statusText: response.statusText,
    headers: response.headers,
  });
};

const { loader, action } = createHybridActionApiRoute(
  {
    body: ChatRequestSchema,
    allowJWT: true,
    authorization: {
      action: "conversation",
    },
    corsStrategy: "all",
  },
  async ({ body, authentication }) => {
    const requestId = crypto.randomUUID();
    const startedAt = Date.now();
    const conversation = await getConversationAndHistory(
      body.id,
      authentication.userId,
    );
    const isAssistantApproval = body.needsApproval;

    const conversationHistory = conversation?.ConversationHistory ?? [];
    const normalizeParts = (parts: any[] | undefined) =>
      (Array.isArray(parts) ? parts : []).filter(Boolean);
    const hasNonEmptyParts = (parts: any[] | undefined) =>
      normalizeParts(parts).length > 0;
    const incomingUserText = body.message?.parts?.[0]?.text;

    if (conversationHistory.length === 1 && !isAssistantApproval && incomingUserText) {
      // Trigger conversation title task
      await enqueueCreateConversationTitle({
        conversationId: body.id,
        message: incomingUserText,
      });
    }

    if (conversationHistory.length > 1 && !isAssistantApproval) {
      const messageParts = body.message?.parts;
      const normalizedMessageParts = normalizeParts(messageParts);

      if (hasNonEmptyParts(normalizedMessageParts)) {
        await upsertConversationHistory(
          body.message?.id ?? crypto.randomUUID(),
          normalizedMessageParts,
          body.id,
          UserTypeEnum.User,
        );
      }
    }

    const messages = conversationHistory.map((history: any) => {
      return {
        parts: normalizeParts(history.parts),
        role:
          history.role ?? (history.userType === "Agent" ? "assistant" : "user"),
        id: history.id,
      };
    });

    const finalFromHistory = messages.filter((m: any) => hasNonEmptyParts(m.parts));
    let finalMessages = finalFromHistory;
    const incomingMessageId = body.message?.id;

    if (!isAssistantApproval) {
      const message = incomingUserText;
      const id = body.message?.id;

      const last = finalFromHistory[finalFromHistory.length - 1];
      const alreadyInHistory = !!(incomingMessageId && last?.id === incomingMessageId);

      if (message && !alreadyInHistory) {
        finalMessages = [
          ...finalFromHistory,
          {
            parts: [{ text: message, type: "text" }],
            role: "user",
            id: id ?? generateId(),
          },
        ];
      }
    } else {
      finalMessages = (body.messages as any[]) ?? [];
      finalMessages = finalMessages
        .map((m: any) => ({
          ...m,
          parts: normalizeParts(m.parts),
        }))
        .filter((m: any) => hasNonEmptyParts(m.parts));
    }

    const validatedMessages = await validateUIMessages({
      messages: finalMessages,
    });

    // If onboarding and no messages yet, use empty messages for agent greeting
    const useEmptyMessages = conversationHistory.length === 0;

    const { systemPrompt, tools, modelMessages } = await buildAgentContext({
      userId: authentication.userId,
      workspaceId: authentication.workspaceId as string,
      conversationId: body.id,
      source: body.source as any,
      finalMessages: useEmptyMessages ? [] : finalMessages,
      preserveToolHistory: Boolean(isAssistantApproval),
    });
    const toolNames = Object.keys(tools ?? {});

    logger.info(
      "Agent chat request prepared",
      {
        conversationId: body.id,
        requestId,
        userId: authentication.userId,
        workspaceId: authentication.workspaceId,
        isAssistantApproval: Boolean(isAssistantApproval),
        source: body.source,
        conversationHistoryCount: conversationHistory.length,
        finalMessageCount: finalMessages.length,
        validatedMessageCount: validatedMessages.length,
        modelMessageCount: modelMessages.length,
        incomingMessageId,
        incomingUserChars: incomingUserText?.length ?? 0,
        toolCount: toolNames.length,
        toolNames,
      },
    );

    const result = await makeModelCall(
      true,
      [
        {
          role: "system",
          content: systemPrompt,
        },
        ...modelMessages,
      ],
      () => {},
      {
        tools,
        stopWhen: [stepCountIs(10)],
        temperature: 0.5,
      },
      "high",
      "core-agent-chat",
      undefined,
      {
        callSite: "core.agent.chat.stream",
        proxyAffinityKey: `conversation:${body.id}`,
      },
    );

    logger.info(
      "Agent chat model stream created",
      {
        conversationId: body.id,
        requestId,
        elapsedMs: Date.now() - startedAt,
      },
    );

    const response = result.toUIMessageStreamResponse({
      generateMessageId: () => crypto.randomUUID(),
      originalMessages: validatedMessages,
      onError: (error: any) => {
        logger.error(
          "Agent chat UI stream error",
          {
            conversationId: body.id,
            requestId,
            elapsedMs: Date.now() - startedAt,
            error,
          },
        );
        return error.message;
      },
      onFinish: async ({ messages }) => {
        const lastMessage = messages.pop();

        logger.info(
          "Agent chat UI stream finish callback",
          {
            conversationId: body.id,
            requestId,
            elapsedMs: Date.now() - startedAt,
            finishedMessageCount: messages.length + (lastMessage ? 1 : 0),
            lastMessageId: lastMessage?.id,
            lastMessageRole: lastMessage?.role,
            lastMessageParts: summarizeParts(lastMessage?.parts),
          },
        );

        if (lastMessage) {
          await upsertConversationHistory(
            lastMessage?.id ?? crypto.randomUUID(),
            lastMessage?.parts,
            body.id,
            UserTypeEnum.Agent,
          );

          // Extract text from message parts and add to queue for ingestion
          const textParts = lastMessage?.parts
            ?.filter((part: any) => part.type === "text" && part.text)
            .map((part: any) => part.text);

          if (textParts && textParts.length > 0) {
            const messageText = textParts.join("\n");

            await addToQueue(
              {
                episodeBody: `<user>${incomingUserText ?? ""}</user><assistant>${messageText}</assistant>`,
                source: "core",
                referenceTime: new Date().toISOString(),
                type: EpisodeType.CONVERSATION,
                sessionId: body.id,
              },
              authentication.userId,
              authentication.workspaceId || "",
            );
          }

          logger.info(
            "Agent chat assistant message persisted",
            {
              conversationId: body.id,
              requestId,
              elapsedMs: Date.now() - startedAt,
              messageId: lastMessage.id,
              textPartCount: textParts?.length ?? 0,
            },
          );
        }
      },
      // async consumeSseStream({ stream }) {
      //   // Create a resumable stream from the SSE stream
      //   const streamContext = createResumableStreamContext({ waitUntil: null });
      //   await streamContext.createNewResumableStream(
      //     conversation.conversationHistoryId,
      //     () => stream,
      //   );
      // },
    });

    return wrapChatResponseForDiagnostics(response, {
      conversationId: body.id,
      requestId,
      startedAt,
    });
  },
);

export { loader, action };
