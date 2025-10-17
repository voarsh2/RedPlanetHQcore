import {
  type LanguageModelV1,
} from "ai";
import { openai } from "@ai-sdk/openai";
import { anthropic } from "@ai-sdk/anthropic";
import { google } from "@ai-sdk/google";
import { createAmazonBedrock } from "@ai-sdk/amazon-bedrock";
import type { OllamaProvider } from "ollama-ai-provider";
import { logger } from "~/services/logger.service";

export interface ProviderContext {
  options?: Record<string, unknown>;
  bedrock: ReturnType<typeof createAmazonBedrock>;
  ollama?: OllamaProvider;
}

export interface ProviderResult {
  instance: LanguageModelV1;
  defaultOptions?: Record<string, unknown>;
}

type ProviderResolver = (
  model: string,
  context: ProviderContext,
) => ProviderResult | undefined;

const OPENAI_MODELS = new Set([
  "gpt-4.1-2025-04-14",
  "gpt-4.1-mini-2025-04-14",
  "gpt-4.1-nano-2025-04-14",
  "gpt-5-2025-08-07",
  "gpt-5-mini-2025-08-07",
]);

const CLAUDE_MODELS = new Set([
  "claude-3-opus-20240229",
  "claude-3-7-sonnet-20250219",
  "claude-3-5-haiku-20241022",
]);

const GEMINI_MODELS = new Set([
  "gemini-2.5-flash-preview-04-17",
  "gemini-2.5-pro-preview-03-25",
  "gemini-2.0-flash",
  "gemini-2.0-flash-lite",
]);

const BEDROCK_MODELS = new Set([
  "us.meta.llama3-3-70b-instruct-v1:0",
  "us.deepseek.r1-v1:0",
  "qwen.qwen3-32b-v1:0",
  "openai.gpt-oss-120b-1:0",
  "us.mistral.pixtral-large-2502-v1:0",
  "us.amazon.nova-premier-v1:0",
]);

const GLM_MODELS = new Set([
  "glm-4",
  "glm-4-air",
  "glm-4-airx",
  "glm-4-flash",
]);

const openAIResolver: ProviderResolver = (model, context) => {
  if (!OPENAI_MODELS.has(model)) {
    return undefined;
  }

  return {
    instance: openai(model, {
      ...context.options,
    }),
    defaultOptions: {
      temperature: 1,
    },
  };
};

const claudeResolver: ProviderResolver = (model, context) => {
  if (!CLAUDE_MODELS.has(model)) {
    return undefined;
  }

  return {
    instance: anthropic(model, {
      ...context.options,
    }),
  };
};

const geminiResolver: ProviderResolver = (model, context) => {
  if (!GEMINI_MODELS.has(model)) {
    return undefined;
  }

  return {
    instance: google(model, {
      ...context.options,
    }),
  };
};

const bedrockResolver: ProviderResolver = (model, context) => {
  if (!BEDROCK_MODELS.has(model)) {
    return undefined;
  }

  return {
    instance: context.bedrock(model),
    defaultOptions: {
      maxTokens: 100000,
    },
  };
};

const glmResolver: ProviderResolver = (model, context) => {
  if (!GLM_MODELS.has(model)) {
    return undefined;
  }

  const apiKey = process.env.GLM_API_KEY;
  if (!apiKey) {
    throw new Error(
      "GLM_API_KEY is required when using a GLM model.",
    );
  }

  const baseURL =
    process.env.GLM_API_BASE_URL ??
    "https://open.bigmodel.cn/api/paas/v4";

  // The GLM REST API follows the OpenAI-compatible schema, so we reuse
  // the OpenAI provider with a custom base URL.
  return {
    instance: openai(model, {
      apiKey,
      baseURL,
      ...context.options,
    }),
  };
};

const ollamaResolver: ProviderResolver = (model, context) => {
  if (!context.ollama) {
    return undefined;
  }

  try {
    return {
      instance: context.ollama(model),
    };
  } catch (error) {
    logger.warn("Failed to resolve model via Ollama provider", {
      error,
      model,
    });
    return undefined;
  }
};

const resolvers: ProviderResolver[] = [
  openAIResolver,
  claudeResolver,
  geminiResolver,
  bedrockResolver,
  glmResolver,
  ollamaResolver,
];

export function resolveProvider(
  model: string,
  context: ProviderContext,
): ProviderResult | undefined {
  for (const resolver of resolvers) {
    const result = resolver(model, context);
    if (result) {
      return result;
    }
  }

  return undefined;
}

export function isKnownModel(model: string) {
  return (
    OPENAI_MODELS.has(model) ||
    CLAUDE_MODELS.has(model) ||
    GEMINI_MODELS.has(model) ||
    BEDROCK_MODELS.has(model) ||
    GLM_MODELS.has(model)
  );
}
