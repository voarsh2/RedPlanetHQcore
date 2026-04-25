import { createHash, randomUUID } from "node:crypto";
import { appendFileSync, mkdirSync } from "node:fs";
import path from "node:path";
import { env } from "~/env.server";
import {
  countTokens,
  shouldApplyContextBudget,
  trimMessagesToBudget,
} from "~/lib/context-budget.server";
import {
  getStoredProxyPreviousResponseId,
  storeProxyPreviousResponseId,
} from "~/lib/openai-proxy-turn-state.server";
import { logger } from "~/services/logger.service";
import type { ModelCallTelemetry } from "~/lib/model.server";

// ---------------------------------------------------------------------------
// Generic wire types
// ---------------------------------------------------------------------------

type ParsedProxyWireBody = Record<string, unknown>;
type ProxyRequestKind = "responses" | "chat_completions";
type ProxyRequestBodyKey = "input" | "messages";

type OpenAIWireRequestContext = {
  url: string;
  method: string;
  requestType: ProxyRequestKind;
  requestBodyKey: ProxyRequestBodyKey;
  isProxyRequest: boolean;
  shouldLogRequest: boolean;
  shouldApplyBudgetTrim: boolean;
  shouldApplyProxyContinuityExperiment: boolean;
};

type BudgetTrimInfo = {
  applied: boolean;
  bodyTokensBefore: number;
  bodyTokensAfter: number;
  bodyCharsBefore: number;
  bodyCharsAfter: number;
  inputCountBefore: number;
  inputCountAfter: number;
  messagesDropped: number;
  messagesTruncated: number;
};

// ---------------------------------------------------------------------------
// Generic wire logging
// ---------------------------------------------------------------------------

function logProxyWireIntercept(context: OpenAIWireRequestContext) {
  logger.info("Proxy wire intercept", {
    url: context.url,
    method: context.method,
    requestType: context.requestType,
    hasOpenAIBaseUrl: !!env.OPENAI_BASE_URL,
    shouldLogRequest: context.shouldLogRequest,
    shouldApplyBudgetTrim: context.shouldApplyBudgetTrim,
    shouldApplyProxyContinuityExperiment:
      context.shouldApplyProxyContinuityExperiment,
  });
}

function logProxyWireBudgetEvaluation(
  context: OpenAIWireRequestContext,
  budgetTrimInfo: BudgetTrimInfo,
  rawBody: string | undefined,
  parsedBody: ParsedProxyWireBody | undefined,
) {
  logger.info("Proxy wire request budget evaluation", {
    requestType: context.requestType,
    configuredBudget: env.LLM_CONTEXT_BUDGET,
    effectiveWireBudget: env.LLM_CONTEXT_BUDGET,
    effectiveWireCharBudget: env.LLM_CONTEXT_BUDGET * 4,
    bodyTokensBefore: budgetTrimInfo.bodyTokensBefore,
    bodyTokensAfter: budgetTrimInfo.bodyTokensAfter,
    bodyCharsBefore: budgetTrimInfo.bodyCharsBefore,
    bodyCharsAfter: budgetTrimInfo.bodyCharsAfter,
    messageArrayKey: context.requestBodyKey,
    inputCountBefore: budgetTrimInfo.inputCountBefore,
    inputCountAfter: budgetTrimInfo.inputCountAfter,
    messagesDropped: budgetTrimInfo.messagesDropped,
    messagesTruncated: budgetTrimInfo.messagesTruncated,
    hasRawBody: !!rawBody,
    bodyParseSucceeded: !!parsedBody,
    model: parsedBody?.model,
    trimmed: budgetTrimInfo.applied,
  });
}

function logProxyWireTrimmed(
  context: OpenAIWireRequestContext,
  budgetTrimInfo: BudgetTrimInfo,
  parsedBody: ParsedProxyWireBody | undefined,
) {
  logger.info("Proxy wire request trimmed to token budget", {
    requestType: context.requestType,
    configuredBudget: env.LLM_CONTEXT_BUDGET,
    effectiveWireBudget: env.LLM_CONTEXT_BUDGET,
    effectiveWireCharBudget: env.LLM_CONTEXT_BUDGET * 4,
    bodyTokensBefore: budgetTrimInfo.bodyTokensBefore,
    bodyTokensAfter: budgetTrimInfo.bodyTokensAfter,
    bodyCharsBefore: budgetTrimInfo.bodyCharsBefore,
    bodyCharsAfter: budgetTrimInfo.bodyCharsAfter,
    messageArrayKey: context.requestBodyKey,
    inputCountBefore: budgetTrimInfo.inputCountBefore,
    inputCountAfter: budgetTrimInfo.inputCountAfter,
    messagesDropped: budgetTrimInfo.messagesDropped,
    messagesTruncated: budgetTrimInfo.messagesTruncated,
    model: parsedBody?.model,
  });
}

function logProxyWireResponseSummary(
  context: OpenAIWireRequestContext,
  response: Response,
  responseBody: string,
  elapsedMs: number,
) {
  const responseContentType = response.headers.get("content-type");
  const responseBodyPreview =
    responseBody.length > 0 ? responseBody.slice(0, 400) : undefined;

  logger.info("Proxy wire response summary", {
    requestType: context.requestType,
    status: response.status,
    ok: response.ok,
    contentType: responseContentType,
    hasBody: responseBody.length > 0,
    responseBodyChars: responseBody.length || 0,
    responseBodyPreview,
    elapsedMs,
  });
}

function getRequestSignal(
  input: RequestInfo | URL,
  init: RequestInit | undefined,
): AbortSignal | undefined {
  if (init?.signal) return init.signal;
  if (input instanceof Request) return input.signal;
  return undefined;
}

// ---------------------------------------------------------------------------
// Claw Bay continuity experiment gate
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
        status?: unknown;
        response?: { id?: unknown; status?: unknown };
      };
      if (parsed.status === "failed" || parsed.response?.status === "failed") {
        return undefined;
      }
      return normalizeProxyValue(parsed.id ?? parsed.response?.id);
    } catch {
      return undefined;
    }
  }

  for (const line of trimmed.split("\n")) {
    if (!line.startsWith("data: ")) continue;
    try {
      const parsed = JSON.parse(line.slice(6)) as {
        response?: { id?: unknown; status?: unknown };
      };
      if (parsed.response?.status === "failed") continue;
      const id = normalizeProxyValue(parsed.response?.id);
      if (id) return id;
    } catch {
      continue;
    }
  }

  return undefined;
}

// ---------------------------------------------------------------------------
// Generic wire request helpers
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

function buildOpenAIWireRequestContext(
  input: RequestInfo | URL,
  init: RequestInit | undefined,
): OpenAIWireRequestContext | null {
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
  const isChatCompletionsRequest =
    method === "POST" && url.includes("/chat/completions");

  if (!isResponsesRequest && !isChatCompletionsRequest) {
    return null;
  }

  const requestType: ProxyRequestKind = isResponsesRequest
    ? "responses"
    : "chat_completions";
  const requestBodyKey: ProxyRequestBodyKey = isChatCompletionsRequest
    ? "messages"
    : "input";
  const isProxyRequest = !!env.OPENAI_BASE_URL;
  const shouldLogRequest = env.LLM_LOG_OPENAI_WIRE;
  const shouldApplyBudgetTrim = isProxyRequest && shouldApplyContextBudget();
  const shouldApplyProxyContinuityExperiment =
    requestType === "responses" &&
    isProxyRequest &&
    shouldEnableProxyContinuityExperiment();

  return {
    url,
    method,
    requestType,
    requestBodyKey,
    isProxyRequest,
    shouldLogRequest,
    shouldApplyBudgetTrim,
    shouldApplyProxyContinuityExperiment,
  };
}

async function extractRawRequestBody(
  input: RequestInfo | URL,
  init: RequestInit | undefined,
): Promise<string | undefined> {
  let rawBody =
    typeof init?.body === "string"
      ? init.body
      : typeof (input as { body?: unknown } | undefined)?.body === "string"
        ? ((input as { body?: string }).body as string)
        : undefined;

  if (!rawBody && input instanceof Request) {
    try {
      const requestText = await input.clone().text();
      rawBody = requestText.length > 0 ? requestText : undefined;
    } catch {
      rawBody = undefined;
    }
  }

  return rawBody;
}

function parseProxyWireBody(
  rawBody: string | undefined,
): ParsedProxyWireBody | undefined {
  if (!rawBody) return undefined;

  try {
    return JSON.parse(rawBody) as ParsedProxyWireBody;
  } catch {
    return undefined;
  }
}

function getRequestMessagesForTrim(
  parsedBody: ParsedProxyWireBody | undefined,
): unknown[] {
  if (Array.isArray(parsedBody?.input)) {
    return parsedBody.input as unknown[];
  }
  if (Array.isArray(parsedBody?.messages)) {
    return parsedBody.messages as unknown[];
  }
  return [];
}

function mergeRequestHeaders(
  headers: RequestInit["headers"],
): Record<string, string> {
  const mergedHeaders: Record<string, string> = {};

  if (headers instanceof Headers) {
    headers.forEach((v, k) => {
      mergedHeaders[k] = v;
    });
  } else if (Array.isArray(headers)) {
    for (const [k, v] of headers) {
      mergedHeaders[k] = v;
    }
  } else if (headers && typeof headers === "object") {
    Object.assign(mergedHeaders, headers);
  }

  return mergedHeaders;
}

// ---------------------------------------------------------------------------
// Claw Bay continuity / cache-affinity helpers
// ---------------------------------------------------------------------------

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

function deriveClawBayEffectiveCacheKey(
  parsedBody: ParsedProxyWireBody | undefined,
  requestBodyKey: ProxyRequestBodyKey,
): string | undefined {
  const promptCacheKey = normalizeProxyValue(parsedBody?.prompt_cache_key);
  if (promptCacheKey) return promptCacheKey;
  if (!parsedBody) return undefined;

  const inputPreview = Array.isArray(parsedBody[requestBodyKey])
    ? JSON.stringify(parsedBody[requestBodyKey]).slice(0, 256)
    : typeof parsedBody.instructions === "string"
      ? parsedBody.instructions.slice(0, 256)
      : "";
  if (!inputPreview) return undefined;

  return hashProxyValue(inputPreview).slice(0, 16);
}

function getClawBayPreviousResponseId(
  parsedBody: ParsedProxyWireBody | undefined,
  effectiveCacheKey: string | undefined,
  shouldApplyProxyContinuityExperiment: boolean,
): string | undefined {
  if (
    !shouldApplyProxyContinuityExperiment ||
    !parsedBody ||
    typeof parsedBody.previous_response_id === "string"
  ) {
    return undefined;
  }

  return getStoredProxyPreviousResponseId(effectiveCacheKey);
}

function applyClawBayRequestMutations(
  parsedBody: ParsedProxyWireBody | undefined,
  effectiveCacheKey: string | undefined,
  previousResponseId: string | undefined,
  shouldApplyProxyContinuityExperiment: boolean,
) {
  if (!parsedBody) return;

  if (previousResponseId) {
    parsedBody.previous_response_id = previousResponseId;
  }

  if (!shouldApplyProxyContinuityExperiment) return;

  if (typeof parsedBody.prompt_cache_key !== "string" && effectiveCacheKey) {
    parsedBody.prompt_cache_key = effectiveCacheKey;
  }
  if (typeof parsedBody.store !== "boolean") {
    parsedBody.store = true;
  }
}

function buildClawBayContinuityHeaders(
  effectiveCacheKey: string | undefined,
  shouldApplyProxyContinuityExperiment: boolean,
): Record<string, string> {
  if (!shouldApplyProxyContinuityExperiment) {
    return {};
  }

  const categorySessionId = getSessionIdForCacheKey(effectiveCacheKey);
  return {
    session_id: categorySessionId,
    "x-client-request-id": categorySessionId,
  };
}

function maybePersistClawBayPreviousResponseId(
  response: Response,
  effectiveCacheKey: string | undefined,
  shouldApplyProxyContinuityExperiment: boolean,
) {
  if (!shouldApplyProxyContinuityExperiment) return;

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

// ---------------------------------------------------------------------------
// Generic wire trimming
// ---------------------------------------------------------------------------

function applyGenericWireBudgetTrim(params: {
  parsedBody: ParsedProxyWireBody | undefined;
  inputItems: unknown[];
  requestBodyKey: ProxyRequestBodyKey;
  shouldApplyBudgetTrim: boolean;
}): { inputItems: unknown[]; budgetTrimInfo?: BudgetTrimInfo } {
  const { parsedBody, requestBodyKey, shouldApplyBudgetTrim } = params;
  let { inputItems } = params;

  if (!shouldApplyBudgetTrim || !parsedBody || inputItems.length === 0) {
    return { inputItems, budgetTrimInfo: undefined };
  }

  const effectiveWireBudget = env.LLM_CONTEXT_BUDGET;
  const effectiveWireCharBudget = effectiveWireBudget * 4;
  let serializedBody = JSON.stringify(parsedBody);
  let bodyTokens = countTokens(serializedBody);
  let bodyChars = serializedBody.length;
  const bodyTokensBefore = bodyTokens;
  const bodyCharsBefore = bodyChars;
  const inputCountBefore = inputItems.length;
  let messagesDropped = 0;
  let messagesTruncated = 0;

  if (bodyTokens > effectiveWireBudget || bodyChars > effectiveWireCharBudget) {
    let working = [...inputItems];
    let lastLength = working.length;
    let lastBodyTokens = bodyTokens;
    let lastBodyChars = bodyChars;

    while (
      working.length > 0 &&
      (bodyTokens > effectiveWireBudget || bodyChars > effectiveWireCharBudget)
    ) {
      const emptyInputTokens = countTokens(
        JSON.stringify({ ...parsedBody, [requestBodyKey]: [] }),
      );
      const remainingInputBudget = Math.max(
        effectiveWireBudget - emptyInputTokens,
        0,
      );
      const result = trimMessagesToBudget(working, remainingInputBudget);

      if (
        result.droppedCount === 0 &&
        result.truncatedCount === 0 &&
        working.length === lastLength
      ) {
        break;
      }

      working = result.messages;
      messagesDropped += result.droppedCount;
      messagesTruncated += result.truncatedCount;
      parsedBody[requestBodyKey] = working;
      serializedBody = JSON.stringify(parsedBody);
      bodyTokens = countTokens(serializedBody);
      bodyChars = serializedBody.length;

      if (
        bodyTokens >= lastBodyTokens &&
        bodyChars >= lastBodyChars &&
        working.length === lastLength
      ) {
        break;
      }

      lastLength = working.length;
      lastBodyTokens = bodyTokens;
      lastBodyChars = bodyChars;
    }

    inputItems = working;
  }

  return {
    inputItems,
    budgetTrimInfo: {
      applied:
        bodyTokensBefore > effectiveWireBudget ||
        bodyCharsBefore > effectiveWireCharBudget,
      bodyTokensBefore,
      bodyTokensAfter: bodyTokens,
      bodyCharsBefore,
      bodyCharsAfter: bodyChars,
      inputCountBefore,
      inputCountAfter: inputItems.length,
      messagesDropped,
      messagesTruncated,
    },
  };
}

// ---------------------------------------------------------------------------
// buildOpenAIWireFetch — single-file generic pipeline + Claw Bay hooks
// ---------------------------------------------------------------------------

export function buildOpenAIWireFetch(baseFetch: typeof fetch): typeof fetch {

  return async (input, init) => {
    const context = buildOpenAIWireRequestContext(input, init);
    if (!context) {
      return baseFetch(input, init);
    }

    logProxyWireIntercept(context);

    if (
      !context.shouldApplyProxyContinuityExperiment &&
      !context.shouldLogRequest &&
      !context.shouldApplyBudgetTrim
    ) {
      return baseFetch(input, init);
    }

    const rawBody = await extractRawRequestBody(input, init);
    const parsedBody = parseProxyWireBody(rawBody);
    const effectiveCacheKey = deriveClawBayEffectiveCacheKey(
      parsedBody,
      context.requestBodyKey,
    );
    const previousResponseId = getClawBayPreviousResponseId(
      parsedBody,
      effectiveCacheKey,
      context.shouldApplyProxyContinuityExperiment,
    );
    const proxyAffinityKey = effectiveCacheKey;
    const proxyAffinityHash = proxyAffinityKey
      ? hashProxyValue(proxyAffinityKey)
      : undefined;
    applyClawBayRequestMutations(
      parsedBody,
      effectiveCacheKey,
      previousResponseId,
      context.shouldApplyProxyContinuityExperiment,
    );

    const trimResult = applyGenericWireBudgetTrim({
      parsedBody,
      inputItems: getRequestMessagesForTrim(parsedBody),
      requestBodyKey: context.requestBodyKey,
      shouldApplyBudgetTrim: context.shouldApplyBudgetTrim,
    });
    const inputItems = trimResult.inputItems;
    const budgetTrimInfo = trimResult.budgetTrimInfo;
    if (budgetTrimInfo) {
      logProxyWireBudgetEvaluation(context, budgetTrimInfo, rawBody, parsedBody);
      if (budgetTrimInfo.applied) {
        logProxyWireTrimmed(context, budgetTrimInfo, parsedBody);
      }
    }

    const serializedBody = JSON.stringify(parsedBody || {});
    const eventId = randomUUID();
    const continuityHeaders = buildClawBayContinuityHeaders(
      effectiveCacheKey,
      context.shouldApplyProxyContinuityExperiment,
    );
    const categorySessionId =
      continuityHeaders.session_id || getSessionIdForCacheKey(effectiveCacheKey);

    if (context.shouldLogRequest) {
      logOpenAIWireEvent({
        event: "request",
        eventId,
        timestamp: new Date().toISOString(),
        url: context.url,
        method: context.method,
        requestType: context.requestType,
        hasRawBody: !!rawBody,
        bodyParseSucceeded: !!parsedBody,
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
        messageArrayKey: context.requestBodyKey,
        bodyCharsBeforeTrim: budgetTrimInfo?.bodyCharsBefore,
        bodyCharsAfterTrim: budgetTrimInfo?.bodyCharsAfter,
        inputCount: inputItems.length,
        contextBudget: env.LLM_CONTEXT_BUDGET,
        budgetTrimApplied: budgetTrimInfo?.applied ?? false,
        bodyTokensBeforeTrim: budgetTrimInfo?.bodyTokensBefore,
        bodyTokensAfterTrim: budgetTrimInfo?.bodyTokensAfter,
        inputCountBeforeTrim: budgetTrimInfo?.inputCountBefore,
        inputCountAfterTrim: budgetTrimInfo?.inputCountAfter,
        budgetMessagesDropped: budgetTrimInfo?.messagesDropped,
        budgetMessagesTruncated: budgetTrimInfo?.messagesTruncated,
        inputPreview: inputItems.slice(0, 3),
        sessionIdHeader: categorySessionId,
        hasSessionIdHeader: Object.keys(continuityHeaders).length > 0,
      });
      writeOpenAIWireBody(eventId, "request", serializedBody);
    }

    const mergedHeaders = mergeRequestHeaders(init?.headers);
    const existingSignal = getRequestSignal(input, init);
    // TODO(upstream-split): proxy-only timeout is private-fork behavior to keep
    // OpenAI-compatible backends from hanging forever. Keep this scoped to proxy
    // mode unless we have upstream evidence that native OpenAI should share it.
    const timeoutSignal = context.isProxyRequest
      ? AbortSignal.timeout(env.OPENAI_PROXY_REQUEST_TIMEOUT_MS)
      : undefined;
    const combinedSignal =
      existingSignal && timeoutSignal
        ? AbortSignal.any([existingSignal, timeoutSignal])
        : existingSignal || timeoutSignal;
    const startedAt = Date.now();

    let response: Response;
    try {
      response = await baseFetch(
        input,
        previousResponseId || rawBody || Object.keys(continuityHeaders).length > 0
          ? {
              ...(init || {}),
              body: serializedBody,
              headers: {
                ...mergedHeaders,
                ...continuityHeaders,
              },
              signal: combinedSignal,
            }
          : {
              ...(init || {}),
              signal: combinedSignal,
            },
      );
    } catch (error) {
      const elapsedMs = Date.now() - startedAt;
      const timedOut =
        !!timeoutSignal &&
        timeoutSignal.aborted &&
        !(existingSignal?.aborted ?? false);

      if (timedOut) {
        logger.error("Proxy wire request timed out", {
          requestType: context.requestType,
          url: context.url,
          method: context.method,
          timeoutMs: env.OPENAI_PROXY_REQUEST_TIMEOUT_MS,
          elapsedMs,
          model: parsedBody?.model,
          bodyChars: serializedBody.length,
          messageArrayKey: context.requestBodyKey,
          inputCount: inputItems.length,
          proxyAffinityHash,
        });
      }

      throw error;
    }

    if (context.shouldApplyProxyContinuityExperiment) {
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
    maybePersistClawBayPreviousResponseId(
      response,
      effectiveCacheKey,
      context.shouldApplyProxyContinuityExperiment,
    );
    let responseBody = "";
    if (context.shouldLogRequest || env.LLM_LOG_OPENAI_WIRE_BODIES) {
      try {
        responseBody = await response.clone().text();
      } catch {
        responseBody = "";
      }
    }
    if (context.shouldLogRequest) {
      logProxyWireResponseSummary(
        context,
        response,
        responseBody,
        Date.now() - startedAt,
      );

      logOpenAIWireEvent({
        event: "response",
        eventId,
        timestamp: new Date().toISOString(),
        url: context.url,
        method: context.method,
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
// Claw Bay prompt-cache / continuity option helpers
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
