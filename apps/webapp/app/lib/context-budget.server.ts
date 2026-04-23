import { encode } from "gpt-tokenizer/encoding/o200k_base";
import { env } from "~/env.server";

// ---------------------------------------------------------------------------
// Token counting
// ---------------------------------------------------------------------------

export function countTokens(text: string): number {
  return encode(text).length;
}

/**
 * Extract text from a message regardless of format.
 * Supports both UI-format ({ parts: [{ type: "text", text: "..." }] })
 * and AI SDK ModelMessage format ({ content: "..." | [{ type: "text", text: "..." }] }).
 */
function extractMessageText(message: unknown): string[] {
  if (!message || typeof message !== "object") return [];
  const msg = message as Record<string, unknown>;

  // UI-format: { parts: [{ type: "text", text: "..." }] }
  if (Array.isArray(msg.parts)) {
    const texts: string[] = [];
    for (const part of msg.parts) {
      if (
        part &&
        typeof part === "object" &&
        (part as { type?: string }).type === "text" &&
        typeof (part as { text?: string }).text === "string"
      ) {
        texts.push((part as { text: string }).text);
      }
    }
    return texts;
  }

  // ModelMessage format: { content: "string" | Array<{ type: "text", text: "..." }> }
  if ("content" in msg) {
    const content = msg.content;
    if (typeof content === "string") return [content];
    if (Array.isArray(content)) {
      const texts: string[] = [];
      for (const part of content) {
        if (
          part &&
          typeof part === "object" &&
          (part as { type?: string }).type === "text" &&
          typeof (part as { text?: string }).text === "string"
        ) {
          texts.push((part as { text: string }).text);
        }
      }
      return texts;
    }
  }

  return [];
}

function countMessageTokens(message: unknown): number {
  const texts = extractMessageText(message);
  let total = 0;
  for (const text of texts) {
    total += countTokens(text);
  }
  return total;
}

function countTotalTokens(messages: unknown[]): number {
  let total = 0;
  for (const msg of messages) {
    total += countMessageTokens(msg);
  }
  return total;
}

// ---------------------------------------------------------------------------
// Truncation helpers
// ---------------------------------------------------------------------------

const TRUNCATION_SUFFIX = "\n\n[Content truncated to fit token budget]";
const MIN_MESSAGES = 2; // Keep at least 1 user + 1 assistant turn

function findLongestMessageIndex(messages: unknown[]): number {
  if (messages.length === 0) return -1;
  let maxTokens = -1;
  let maxIdx = -1;
  for (let i = 0; i < messages.length; i++) {
    const tokens = countMessageTokens(messages[i]);
    if (tokens > maxTokens) {
      maxTokens = tokens;
      maxIdx = i;
    }
  }
  return maxIdx;
}

type TextAccessor = {
  getText: () => string;
  setText: (text: string) => void;
};

/**
 * Get a mutable accessor for the first text part in a message.
 * Works with both UI-format and ModelMessage format.
 */
function getFirstTextAccessor(message: unknown): TextAccessor | null {
  if (!message || typeof message !== "object") return null;
  const msg = message as Record<string, unknown>;

  // UI-format
  if (Array.isArray(msg.parts)) {
    for (const part of msg.parts) {
      if (
        part &&
        typeof part === "object" &&
        (part as { type?: string }).type === "text" &&
        typeof (part as { text?: string }).text === "string"
      ) {
        const textPart = part as { text: string };
        return {
          getText: () => textPart.text,
          setText: (text: string) => {
            textPart.text = text;
          },
        };
      }
    }
    return null;
  }

  // ModelMessage format
  if ("content" in msg) {
    const content = msg.content;
    if (typeof content === "string") {
      return {
        getText: () => msg.content as string,
        setText: (text: string) => {
          msg.content = text;
        },
      };
    }
    if (Array.isArray(content)) {
      for (const part of content) {
        if (
          part &&
          typeof part === "object" &&
          (part as { type?: string }).type === "text" &&
          typeof (part as { text?: string }).text === "string"
        ) {
          const textPart = part as { text: string };
          return {
            getText: () => textPart.text,
            setText: (text: string) => {
              textPart.text = text;
            },
          };
        }
      }
    }
  }

  return null;
}

/**
 * Truncate text to fit within a token budget using a character approximation.
 * ~4 chars/token is a safe heuristic for o200k encoding — tends to over-truncate,
 * which is safer than going over budget.
 */
function truncateTextToTokenBudget(text: string, maxTokens: number): string {
  const charBudget = Math.max(maxTokens * 4, 100);
  if (text.length <= charBudget) return text;

  let truncated = text.slice(0, charBudget);
  const lastSpace = truncated.lastIndexOf(" ");
  if (lastSpace > charBudget * 0.8) {
    truncated = truncated.slice(0, lastSpace);
  }
  return truncated;
}

// ---------------------------------------------------------------------------
// Main trim function
// ---------------------------------------------------------------------------

export function trimMessagesToBudget(
  messages: unknown[],
  budget: number,
): {
  messages: unknown[];
  droppedCount: number;
  truncatedCount: number;
  totalTokens: number;
} {
  if (messages.length === 0 || budget <= 0) {
    return { messages, droppedCount: 0, truncatedCount: 0, totalTokens: 0 };
  }

  let working = [...messages];
  let droppedCount = 0;

  // Phase 1: Drop oldest messages first
  while (countTotalTokens(working) > budget && working.length > MIN_MESSAGES) {
    working.shift();
    droppedCount++;
  }

  let totalTokens = countTotalTokens(working);

  // Phase 2: Truncate individual messages if still over budget
  let truncatedCount = 0;
  while (totalTokens > budget) {
    const longestIdx = findLongestMessageIndex(working);
    if (longestIdx === -1) break;

    const textAccessor = getFirstTextAccessor(working[longestIdx]);
    if (!textAccessor) break;

    const suffixTokens = countTokens(TRUNCATION_SUFFIX);
    const currentTotal = totalTokens;
    const currentText = textAccessor.getText();
    const withoutThisPart = currentTotal - countTokens(currentText);
    const availableTokens = budget - withoutThisPart;
    const maxTextTokens = Math.max(availableTokens - suffixTokens, 10);

    let nextText = truncateTextToTokenBudget(currentText, maxTextTokens);
    if (!nextText.endsWith(TRUNCATION_SUFFIX.trim())) {
      nextText += TRUNCATION_SUFFIX;
    }
    textAccessor.setText(nextText);

    truncatedCount++;
    totalTokens = countTotalTokens(working);
    if (totalTokens <= budget) break;
  }

  return {
    messages: working,
    droppedCount,
    truncatedCount,
    totalTokens,
  };
}

// ---------------------------------------------------------------------------
// Gate: only apply when proxy is active AND budget is set
// ---------------------------------------------------------------------------

export function shouldApplyContextBudget(): boolean {
  return !!(env.LLM_CONTEXT_BUDGET && env.OPENAI_BASE_URL);
}
