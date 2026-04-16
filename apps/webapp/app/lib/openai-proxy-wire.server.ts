import { createHash, randomUUID } from "node:crypto";
import { appendFileSync, mkdirSync } from "node:fs";
import path from "node:path";
import { env } from "~/env.server";
import {
  shouldApplyContextBudget,
  trimMessagesToBudget,
} from "~/lib/context-budget.server";
import {
  getStoredProxyPreviousResponseId,
  storeProxyPreviousResponseId,
} from "~/lib/openai-proxy-turn-state.server";
import type { ModelCallTelemetry } from "~/lib/model.server";

// ---------------------------------------------------------------------------
// Proxy continuity experiment gate
// ---------------------------------------------------------------------------

function shouldEnableProxyContinuityExperiment(): boolean {
  return !!env.OPENAI_BASE_URL && env.OPENAI_PROXY_ENABLE_CONTINUITY_EXPERIMENT;
}

// ---------------------------------------------------------------------------
// Wire logging (gated by LLM_LOG_OPENAI_WIRE / LLM_LOG_OPENAI_WIRE_BODIES)
// ---------------------------------------------------------------------------

function logOpenAIWireEvent(event: Record<string, unknown>) {
  if (!env.LLM_LOG_OPENAI_WIRE) return;

  const logDir = "/tmp/core-openai-wire";
  mkdirSync(logDir, { recursive: true });
  appendFileSync(
    path.join(logDir, "responses-wire.jsonl"),
    `${JSON.stringify(event)}\n`,
    "utf8",
  );
}

function writeOpenAIWireBody(
  eventId: string,
  phase: "request" | "response",
  body: string,
) {
  if (!env.LLM_LOG_OPENAI_WIRE_BODIES) return;

  const logDir = "/tmp/core-openai-wire/bodies";
  mkdirSync(logDir, { recursive: true });
  appendFileSync(path.join(logDir, `${eventId}.${phase}.json`), body, "utf8");
}

// ---------------------------------------------------------------------------
// Proxy value helpers
// ---------------------------------------------------------------------------

function normalizeProxyValue(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const trimmed = value.trim();
  if (!trimmed) return undefined;
  return trimmed.slice(0, 256);
}

function hashProxyValue(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function extractProxyResponseId(body: string): string | undefined {
  const trimmed = body.trim();
  if (!trimmed) return undefined;

  if (trimmed.startsWith("{")) {
    try {
      const parsed = JSON.parse(trimmed) as {
        id?: unknown;
        response?: { id?: unknown };
      };
      return normalizeProxyValue(parsed.id ?? parsed.response?.id);
    } catch {
      return undefined;
    }
  }

  for (const line of trimmed.split("\n")) {
    if (!line.startsWith("data: ")) continue;
    try {
      const parsed = JSON.parse(line.slice(6)) as {
        response?: { id?: unknown };
      };
      const id = normalizeProxyValue(parsed.response?.id);
      if (id) return id;
    } catch {
      continue;
    }
  }

  return undefined;
}

// ---------------------------------------------------------------------------
// buildOpenAIWireFetch — intercepts /responses requests to the proxy
// ---------------------------------------------------------------------------

function extractUrlString(input: unknown): string {
  if (typeof input === "string") return input;
  if (input instanceof URL) return input.toString();
  if (
    input &&
    typeof input === "object" &&
    "url" in input &&
    typeof (input as { url?: unknown }).url === "string"
  ) {
    return (input as { url: string }).url;
  }
  return "";
}

// Stable session identifiers per request category — used as session_id
// header so Claw Bay routes requests to consistent backends.
// Separate IDs per category prevent concurrent requests from racing
// on the same prompt_cache_key namespace (which causes cache invalidation).
const PROXY_SESSION_IDS: Record<string, string> = {
  conversation: randomUUID(),
  "session-compaction": randomUUID(),
  normalization: randomUUID(),
  "label-extraction": randomUUID(),
  "combined-extraction": randomUUID(),
  default: randomUUID(),
};

function getSessionIdForCacheKey(
  cacheKey: string | undefined,
): string {
  if (!cacheKey) return PROXY_SESSION_IDS.default;

  // conversation:* keys → chat session
  if (cacheKey.startsWith("conversation:")) return PROXY_SESSION_IDS.conversation;

  // Known job key patterns (may be prefixed with workspace:uuid:)
  if (cacheKey.includes("normalization")) return PROXY_SESSION_IDS.normalization;
  if (cacheKey.includes("label-extraction")) return PROXY_SESSION_IDS["label-extraction"];
  if (cacheKey.includes("combined-extraction")) return PROXY_SESSION_IDS["combined-extraction"];
  if (cacheKey.includes("session-compaction")) return PROXY_SESSION_IDS["session-compaction"];

  // Hash-derived keys (16-char hex) → default session
  return PROXY_SESSION_IDS.default;
}

export function buildOpenAIWireFetch(baseFetch: typeof fetch): typeof fetch {

  return async (input, init) => {
    const url = extractUrlString(input);
    const inputMethod =
      input &&
      typeof input === "object" &&
      "method" in input &&
      typeof (input as { method?: unknown }).method === "string"
        ? (input as { method: string }).method
        : undefined;
    const method = (init?.method || inputMethod || "GET").toUpperCase();
    const isResponsesRequest = method === "POST" && url.includes("/responses");
    const isProxyResponsesRequest = isResponsesRequest && !!env.OPENAI_BASE_URL;
    const shouldApplyProxyContinuityExperiment =
      isProxyResponsesRequest && shouldEnableProxyContinuityExperiment();
    const shouldLogRequest = env.LLM_LOG_OPENAI_WIRE && isResponsesRequest;

    if (!shouldApplyProxyContinuityExperiment && !shouldLogRequest) {
      return baseFetch(input, init);
    }

    let parsedBody: Record<string, unknown> | undefined;
    const rawBody =
      typeof init?.body === "string"
        ? init.body
        : typeof (input as { body?: unknown } | undefined)?.body === "string"
          ? ((input as { body?: string }).body as string)
          : undefined;

    if (rawBody) {
      try {
        parsedBody = JSON.parse(rawBody) as Record<string, unknown>;
      } catch {
        parsedBody = undefined;
      }
    }

    const promptCacheKey = normalizeProxyValue(parsedBody?.prompt_cache_key);
    // Derive a stable hash from the first ~200 chars of the prompt when no
    // prompt_cache_key is present — gives Claw Bay something sticky to route on.
    let effectiveCacheKey = promptCacheKey;
    if (!effectiveCacheKey && parsedBody) {
      const inputPreview = Array.isArray(parsedBody.input)
        ? JSON.stringify(parsedBody.input).slice(0, 256)
        : typeof parsedBody.instructions === "string"
          ? parsedBody.instructions.slice(0, 256)
          : "";
      if (inputPreview) {
        effectiveCacheKey = hashProxyValue(inputPreview).slice(0, 16);
      }
    }

    // TODO(upstream-split): replaying previous_response_id is proxy-specific continuity
    // glue. Keep this separate from the broadly portable prompt-trimming work.
    const previousResponseId =
      shouldApplyProxyContinuityExperiment &&
      parsedBody &&
      typeof parsedBody.previous_response_id !== "string"
        ? getStoredProxyPreviousResponseId(effectiveCacheKey)
        : undefined;
    const proxyAffinityKey = effectiveCacheKey;
    const proxyAffinityHash = proxyAffinityKey
      ? hashProxyValue(proxyAffinityKey)
      : undefined;

    if (previousResponseId && parsedBody) {
      parsedBody.previous_response_id = previousResponseId;
    }

    // Ensure prompt_cache_key is present on the wire — the Vercel AI SDK sets it
    // via providerOptions, but we reinforce it here so Claw Bay always sees it.
    if (shouldApplyProxyContinuityExperiment && parsedBody) {
      if (typeof parsedBody.prompt_cache_key !== "string" && effectiveCacheKey) {
        parsedBody.prompt_cache_key = effectiveCacheKey;
      }
      // Explicitly enable storage/caching — some proxies disable caching without it.
      if (typeof parsedBody.store !== "boolean") {
        parsedBody.store = true;
      }
    }

    let inputItems = Array.isArray(parsedBody?.input)
      ? (parsedBody.input as unknown[])
      : [];

    // Hard cap on request size at the proxy wire layer — trims oldest messages
    // when the input exceeds LLM_CONTEXT_BUDGET. Prevents the proxy from
    // silently rejecting oversized requests (empty stream).
    if (shouldApplyContextBudget() && parsedBody && inputItems.length > 0) {
      const result = trimMessagesToBudget(inputItems, env.LLM_CONTEXT_BUDGET);
      if (result.droppedCount > 0 || result.truncatedCount > 0) {
        parsedBody.input = result.messages;
        inputItems = result.messages;
      }
    }

    const serializedBody = JSON.stringify(parsedBody || {});
    const eventId = randomUUID();

    // Build headers for sticky session routing on Claw Bay.
    // For codex-native traffic: session_id header routes to the same backend.
    // For OpenAI-compatible traffic: prompt_cache_key in the body handles routing.
    // Use a category-specific session ID to prevent concurrent requests with
    // different prompt_cache_keys from racing on the same backend.
    const categorySessionId = getSessionIdForCacheKey(promptCacheKey);
    const continuityHeaders: Record<string, string> =
      shouldApplyProxyContinuityExperiment
        ? {
            session_id: categorySessionId,
            "x-client-request-id": categorySessionId,
          }
        : {};

    if (shouldLogRequest) {
      logOpenAIWireEvent({
        event: "request",
        eventId,
        timestamp: new Date().toISOString(),
        url,
        method,
        bodyHash: createHash("sha256").update(serializedBody).digest("hex"),
        bodyChars: serializedBody.length,
        hasPromptCacheKey: typeof parsedBody?.prompt_cache_key === "string",
        promptCacheKey: parsedBody?.prompt_cache_key,
        promptCacheRetention: parsedBody?.prompt_cache_retention,
        store: parsedBody?.store,
        stream: parsedBody?.stream,
        model: parsedBody?.model,
        hasInstructions: typeof parsedBody?.instructions === "string",
        instructionsChars:
          typeof parsedBody?.instructions === "string"
            ? parsedBody.instructions.length
            : 0,
        toolsCount: Array.isArray(parsedBody?.tools) ? parsedBody.tools.length : 0,
        toolChoice: parsedBody?.tool_choice,
        parallelToolCalls: parsedBody?.parallel_tool_calls,
        serviceTier: parsedBody?.service_tier,
        proxyAffinityKey,
        proxyAffinityHash,
        previousResponseIdHash: previousResponseId
          ? hashProxyValue(previousResponseId)
          : undefined,
        reasoning: parsedBody?.reasoning,
        text: parsedBody?.text,
        include: parsedBody?.include,
        inputCount: inputItems.length,
        inputPreview: inputItems.slice(0, 3),
        sessionIdHeader: categorySessionId,
        hasSessionIdHeader: Object.keys(continuityHeaders).length > 0,
      });
      writeOpenAIWireBody(eventId, "request", serializedBody);
    }

    // Merge headers safely — init.headers can be Headers, string[][], or Record.
    const existingHeaders = init?.headers;
    const mergedHeaders: Record<string, string> = {};
    if (existingHeaders instanceof Headers) {
      existingHeaders.forEach((v, k) => {
        mergedHeaders[k] = v;
      });
    } else if (Array.isArray(existingHeaders)) {
      for (const [k, v] of existingHeaders) {
        mergedHeaders[k] = v;
      }
    } else if (existingHeaders && typeof existingHeaders === "object") {
      Object.assign(mergedHeaders, existingHeaders);
    }

    const response = await baseFetch(
      input,
      previousResponseId || rawBody || Object.keys(continuityHeaders).length > 0
        ? {
            ...(init || {}),
            body: serializedBody,
            headers: {
              ...mergedHeaders,
              ...continuityHeaders,
            },
          }
        : init,
    );

    // Debug: log what headers were actually sent
    if (shouldApplyProxyContinuityExperiment) {
      logOpenAIWireEvent({
        event: "headers-debug",
        eventId,
        mergedHeaders,
        continuityHeaders,
        finalHeaders: previousResponseId || rawBody || Object.keys(continuityHeaders).length > 0
          ? { ...mergedHeaders, ...continuityHeaders }
          : "unchanged",
      });
    }
    if (shouldApplyProxyContinuityExperiment) {
      void response
        .clone()
        .text()
        .then((body) => {
          storeProxyPreviousResponseId(
            effectiveCacheKey,
            extractProxyResponseId(body),
          );
        })
        .catch(() => {});
    }
    let responseBody = "";
    if (env.LLM_LOG_OPENAI_WIRE_BODIES) {
      try {
        responseBody = await response.clone().text();
      } catch {
        responseBody = "";
      }
    }
    if (shouldLogRequest) {
      logOpenAIWireEvent({
        event: "response",
        eventId,
        timestamp: new Date().toISOString(),
        url,
        method,
        status: response.status,
        ok: response.ok,
      });
      if (responseBody) {
        writeOpenAIWireBody(eventId, "response", responseBody);
      }
    }
    return response;
  };
}

// ---------------------------------------------------------------------------
// buildProxyPromptCacheKey — overrides cache key for proxy continuity
// ---------------------------------------------------------------------------

function buildProxyPromptCacheKey(
  cacheKey: string | undefined,
  telemetry: ModelCallTelemetry | undefined,
): string | undefined {
  const normalizedCacheKey = normalizeProxyValue(cacheKey);

  return normalizeProxyValue(
    telemetry?.proxyAffinityKey || normalizedCacheKey || telemetry?.callSite,
  );
}

// ---------------------------------------------------------------------------
// buildOpenAIPromptCacheOptions — providerOptions for prompt caching
// ---------------------------------------------------------------------------

export type OpenAIApiMode = "responses" | "chat_completions";

export function buildOpenAIPromptCacheOptions({
  model,
  cacheKey,
  reasoningEffort,
  configuredOpenaiApiMode,
  effectiveOpenaiApiMode,
  useOllamaForChat,
  telemetry,
}: {
  model: string;
  cacheKey?: string;
  reasoningEffort?: "low" | "medium" | "high";
  configuredOpenaiApiMode: OpenAIApiMode;
  effectiveOpenaiApiMode: OpenAIApiMode;
  useOllamaForChat: boolean;
  telemetry?: ModelCallTelemetry;
}): {
  providerOptions?: { openai: Record<string, unknown> };
  promptCacheConfigured: boolean;
  promptCacheStrategy: string;
} {
  if (!model.includes("gpt") || useOllamaForChat) {
    return {
      promptCacheConfigured: false,
      promptCacheStrategy: "disabled",
    };
  }

  const isResponsesMode = effectiveOpenaiApiMode === "responses";
  const isForcedProxyResponses =
    configuredOpenaiApiMode === "chat_completions" &&
    effectiveOpenaiApiMode === "responses" &&
    !!env.OPENAI_BASE_URL;
  const isProxyResponsesMode = isResponsesMode && !!env.OPENAI_BASE_URL;

  if (!isResponsesMode) {
    return {
      promptCacheConfigured: false,
      promptCacheStrategy: "unsupported-mode",
    };
  }

  const openaiOptions: Record<string, unknown> = {
    promptCacheKey: cacheKey || `model-call`,
  };

  if (isResponsesMode) {
    if (isProxyResponsesMode && shouldEnableProxyContinuityExperiment()) {
      const proxyPromptCacheKey = buildProxyPromptCacheKey(cacheKey, telemetry);
      if (proxyPromptCacheKey) {
        openaiOptions.promptCacheKey = proxyPromptCacheKey;
      }
    }
  }

  if (model.startsWith("gpt-5")) {
    if (model.includes("mini")) {
      if (isResponsesMode) {
        openaiOptions.reasoningEffort = "low";
      }
    } else {
      openaiOptions.promptCacheRetention = "24h";
      if (isResponsesMode) {
        openaiOptions.reasoningEffort = reasoningEffort || "none";
      }
    }
  }

  return {
    providerOptions: {
      openai: openaiOptions,
    },
    promptCacheConfigured: true,
    promptCacheStrategy: isProxyResponsesMode
      ? isForcedProxyResponses
        ? "responses-proxy-forced"
        : "responses-proxy"
      : "responses-native",
  };
}
