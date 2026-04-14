#!/usr/bin/env node

function repeatText(label, size) {
  const seed = `${label} `;
  return seed.repeat(Math.ceil(size / seed.length)).slice(0, size);
}

function hasApprovalStateDeep(parts) {
  for (const part of parts) {
    if (!part || typeof part !== "object") continue;
    if (
      part.state === "approval-requested" ||
      part.state === "approval-responded"
    ) {
      return true;
    }

    const nestedParts = Array.isArray(part.output?.parts)
      ? part.output.parts
      : Array.isArray(part.output?.content)
        ? part.output.content
        : [];

    if (nestedParts.length > 0 && hasApprovalStateDeep(nestedParts)) {
      return true;
    }
  }

  return false;
}

function getTopLevelTextParts(parts) {
  return parts
    .filter((part) => part?.type === "text" && typeof part.text === "string")
    .map((part) => ({ type: "text", text: part.text.trim() }))
    .filter((part) => part.text.length > 0);
}

function getToolSummary(parts) {
  const tools = parts
    .filter((part) => typeof part?.type === "string" && part.type.includes("tool-"))
    .map((part) => {
      const toolName = String(part.type).replace("tool-", "");
      const state =
        typeof part.state === "string" && part.state.length > 0
          ? ` (${part.state})`
          : "";
      return `${toolName}${state}`;
    });

  if (tools.length === 0) return null;

  return `Prior tool activity omitted for brevity: ${tools.join(", ")}.`;
}

function sanitizeMessageForModelContext(message) {
  const normalizedParts = Array.isArray(message?.parts)
    ? message.parts.filter(Boolean)
    : [];

  if (normalizedParts.length === 0) return message;

  const textParts = getTopLevelTextParts(normalizedParts);
  if (textParts.length > 0) {
    return { ...message, parts: textParts };
  }

  const toolSummary = getToolSummary(normalizedParts);
  if (toolSummary) {
    return {
      ...message,
      parts: [{ type: "text", text: toolSummary }],
    };
  }

  return { ...message, parts: [] };
}

function sanitizeMessagesForModelContext(messages, preserveToolHistory = false) {
  if (preserveToolHistory) return messages;

  return messages
    .map((message) => {
      const normalizedParts = Array.isArray(message?.parts)
        ? message.parts.filter(Boolean)
        : [];

      if (
        message?.role === "assistant" &&
        normalizedParts.length > 0 &&
        !hasApprovalStateDeep(normalizedParts)
      ) {
        return sanitizeMessageForModelContext(message);
      }

      if (message?.role === "user" && normalizedParts.length > 0) {
        return sanitizeMessageForModelContext(message);
      }

      return message;
    })
    .filter((message) => Array.isArray(message?.parts) && message.parts.length > 0);
}

function applyCompactedSessionWindow(
  messages,
  compactSummary,
  recentWindow = 8,
) {
  if (!compactSummary || messages.length <= recentWindow) {
    return messages;
  }

  return [
    {
      id: "session-summary",
      role: "assistant",
      parts: [
        {
          type: "text",
          text: `Earlier session summary: ${compactSummary}`,
        },
      ],
    },
    ...messages.slice(-recentWindow),
  ];
}

function createHeavyAssistantMessage(id, { includeTopLevelText = true } = {}) {
  const parts = [];

  if (includeTopLevelText) {
    parts.push({
      type: "text",
      text: repeatText(
        "Checked memory, integrations, and web. Here is the synthesized answer with tradeoffs, findings, and next steps.",
        1800,
      ),
    });
  }

  parts.push({
    type: "tool-gather_context",
    state: "output-available",
    input: {
      query: "Find prior discussions, related issues, and external docs.",
    },
    output: {
      parts: [
        {
          type: "text",
          text: repeatText("memory-episode", 32000),
        },
        {
          type: "tool-memory_search",
          state: "output-available",
          input: {
            query: "core project auth proxy and prompt cache decisions",
          },
          output: {
            content: [
              {
                type: "text",
                text: repeatText("memory-facts", 26000),
              },
            ],
          },
        },
        {
          type: "tool-integration_query",
          state: "output-available",
          input: {
            integration: "github",
            query: "open PRs and issues about auth proxy",
          },
          output: {
            parts: [
              {
                type: "text",
                text: repeatText("integration-json", 28000),
              },
            ],
          },
        },
      ],
    },
  });

  parts.push({
    type: "tool-web_search",
    state: "output-available",
    input: { query: "latest openai compatible prompt cache behavior" },
    output: {
      content: [
        {
          type: "text",
          text: repeatText("web-results", 18000),
        },
      ],
    },
  });

  return {
    id,
    role: "assistant",
    parts,
  };
}

function buildScenario(name, assistantTurns) {
  const messages = [];

  for (let index = 0; index < assistantTurns; index += 1) {
    messages.push({
      id: `user-${index + 1}`,
      role: "user",
      parts: [
        {
          type: "text",
          text: repeatText(
            `Question ${index + 1}: explain the latest findings about the core project prompt cache issue, compare proxy and direct OpenAI behavior, and outline the next debugging steps. `,
            2200,
          ),
        },
      ],
    });
    messages.push(createHeavyAssistantMessage(`assistant-${index + 1}`));
  }

  messages.push({
    id: `user-final`,
    role: "user",
    parts: [
      {
        type: "text",
        text: repeatText(
          "Can you keep digging and explain what is causing the meg prompt growth, which layers are replaying context, and what measurable proof we should gather next? ",
          1800,
        ),
      },
    ],
  });

  return { name, messages };
}

function metricsForMessages(messages) {
  const json = JSON.stringify(messages);
  const totalParts = messages.reduce(
    (sum, message) => sum + (Array.isArray(message.parts) ? message.parts.length : 0),
    0,
  );

  return {
    messageCount: messages.length,
    topLevelPartCount: totalParts,
    jsonChars: json.length,
    approxTokens: Math.ceil(json.length / 4),
  };
}

const scenarios = [
  buildScenario("single-heavy-turn", 1),
  buildScenario("three-heavy-turns", 3),
  buildScenario("five-heavy-turns", 5),
];

const results = scenarios.map((scenario) => {
  const sanitized = sanitizeMessagesForModelContext(scenario.messages, false);
  const compacted = applyCompactedSessionWindow(
    sanitized,
    repeatText("session-summary", 1200),
    8,
  );
  const rawMetrics = metricsForMessages(scenario.messages);
  const sanitizedMetrics = metricsForMessages(sanitized);
  const compactedMetrics = metricsForMessages(compacted);

  return {
    scenario: scenario.name,
    raw: rawMetrics,
    sanitized: sanitizedMetrics,
    compacted: compactedMetrics,
    reduction: {
      sanitized: {
        jsonChars: rawMetrics.jsonChars - sanitizedMetrics.jsonChars,
        approxTokens: rawMetrics.approxTokens - sanitizedMetrics.approxTokens,
        percent:
          rawMetrics.jsonChars === 0
            ? 0
            : Number(
                (
                  ((rawMetrics.jsonChars - sanitizedMetrics.jsonChars) /
                    rawMetrics.jsonChars) *
                  100
                ).toFixed(2),
              ),
      },
      compacted: {
        jsonChars: rawMetrics.jsonChars - compactedMetrics.jsonChars,
        approxTokens: rawMetrics.approxTokens - compactedMetrics.approxTokens,
        percent:
          rawMetrics.jsonChars === 0
            ? 0
            : Number(
                (
                  ((rawMetrics.jsonChars - compactedMetrics.jsonChars) /
                    rawMetrics.jsonChars) *
                  100
                ).toFixed(2),
              ),
      },
    },
  };
});

console.log(JSON.stringify({ scenarios: results }, null, 2));
