const PROXY_STATE_TTL_MS = 30 * 60 * 1000;
const PROXY_STATE_MAX_ENTRIES = 256;

// TODO(upstream-split): This file is specific to our proxy-cache experiment.
// Keep generic prompt-size reductions separate if we backport upstream.
type ProxyStateEntry = {
  value: string;
  updatedAt: number;
};

const promptCachePreviousResponseIds = new Map<string, ProxyStateEntry>();

function normalize(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  return trimmed ? trimmed.slice(0, 512) : undefined;
}

function isConversationScopedPromptCacheKey(promptCacheKey: string | undefined): boolean {
  return !!promptCacheKey && promptCacheKey.startsWith("conversation:");
}

function purgeExpiredEntries(now = Date.now()) {
  for (const [key, entry] of promptCachePreviousResponseIds.entries()) {
    if (now - entry.updatedAt > PROXY_STATE_TTL_MS) {
      promptCachePreviousResponseIds.delete(key);
    }
  }
}

export function getStoredProxyPreviousResponseId(
  promptCacheKey: string | undefined,
): string | undefined {
  const normalizedKey = normalize(promptCacheKey);
  if (!isConversationScopedPromptCacheKey(normalizedKey)) return undefined;

  purgeExpiredEntries();
  const entry = normalizedKey
    ? promptCachePreviousResponseIds.get(normalizedKey)
    : undefined;
  return entry?.value;
}

export function storeProxyPreviousResponseId(
  promptCacheKey: string | undefined,
  responseId: string | undefined,
): void {
  const normalizedKey = normalize(promptCacheKey);
  const normalizedResponseId = normalize(responseId);
  if (!isConversationScopedPromptCacheKey(normalizedKey) || !normalizedResponseId) return;

  purgeExpiredEntries();
  if (promptCachePreviousResponseIds.size >= PROXY_STATE_MAX_ENTRIES) {
    const oldestKey = promptCachePreviousResponseIds.keys().next().value;
    if (typeof oldestKey === "string") {
      promptCachePreviousResponseIds.delete(oldestKey);
    }
  }

  promptCachePreviousResponseIds.set(normalizedKey, {
    value: normalizedResponseId,
    updatedAt: Date.now(),
  });
}
