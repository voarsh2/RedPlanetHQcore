#!/usr/bin/env node

import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_ENV_FILES = [
  path.resolve(SCRIPT_DIR, "../../../hosting/docker/.env"),
  path.resolve(SCRIPT_DIR, "../../../hosting/docker/.env.local"),
];

function parseEnvFile(filePath) {
  if (!existsSync(filePath)) return {};

  const parsed = {};
  const content = readFileSync(filePath, "utf8");

  for (const rawLine of content.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;

    const separatorIndex = line.indexOf("=");
    if (separatorIndex === -1) continue;

    const key = line.slice(0, separatorIndex).trim();
    let value = line.slice(separatorIndex + 1).trim();

    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }

    parsed[key] = value;
  }

  return parsed;
}

function loadEffectiveEnv() {
  const requestedEnvFile = process.argv
    .find((arg) => arg.startsWith("--env-file="))
    ?.split("=")[1];

  const merged = {};
  const files = requestedEnvFile
    ? [path.resolve(process.cwd(), requestedEnvFile)]
    : DEFAULT_ENV_FILES;

  for (const filePath of files) {
    Object.assign(merged, parseEnvFile(filePath));
  }

  return {
    ...merged,
    ...process.env,
    __files: files.filter((filePath) => existsSync(filePath)),
  };
}

const EFFECTIVE_ENV = loadEffectiveEnv();
const DEFAULT_MODEL = EFFECTIVE_ENV.MODEL || "gpt-4.1-mini";
const DEFAULT_BASE_URL = EFFECTIVE_ENV.OPENAI_BASE_URL || "https://api.openai.com/v1";
const DEFAULT_API_MODE =
  EFFECTIVE_ENV.OPENAI_API_MODE === "responses" ? "responses" : "chat_completions";

const DEFAULT_INTEGRATIONS = [
  "1. **GitHub** (Account ID: github-acct-1) (Identifier: voarsh)",
  "2. **Slack** (Account ID: slack-acct-1) (Identifier: core-workspace)",
  "3. **Gmail** (Account ID: gmail-acct-1) (Identifier: reese@example.com)",
].join("\n");

const DEFAULT_GATEWAYS = [
  "1. **Harshith Mac** (tool: gateway_harshith_mac): personal coding tasks",
  "2. **Browser Agent** (tool: gateway_browser_agent): browser automation and screenshots",
].join("\n");

const DEFAULT_PERSONA = `USER PERSONA (identity, preferences, directives - use this FIRST before searching memory):
Reese
- prefers practical experiments over theory
- keep scope narrow and least-invasive
- currently evaluating the core project private fork for self-host / proxy behavior`;
const DEFAULT_PAD_CHARS = Number(
  process.argv
    .find((arg) => arg.startsWith("--pad-static-chars="))
    ?.split("=")[1] || "0",
);
const USE_CACHE_HINTS = process.argv.includes("--with-cache-hints");
const TEMPLATE_BODY_PATH = process.argv
  .find((arg) => arg.startsWith("--template-body-path="))
  ?.split("=")[1];
const FRESH_CACHE_KEY = process.argv.includes("--fresh-cache-key");

function repeatText(label, size) {
  const seed = `${label} `;
  return seed.repeat(Math.ceil(size / seed.length)).slice(0, size);
}

function serializePromptMetrics(prompt) {
  return {
    chars: prompt.length,
    hash: createHash("sha256").update(prompt).digest("hex"),
    prefix512Hash: createHash("sha256")
      .update(prompt.slice(0, 512))
      .digest("hex"),
    prefix2048Hash: createHash("sha256")
      .update(prompt.slice(0, 2048))
      .digest("hex"),
  };
}

function sharedPrefixChars(left, right) {
  const limit = Math.min(left.length, right.length);
  let index = 0;

  while (index < limit && left[index] === right[index]) {
    index += 1;
  }

  return index;
}

function toLegacyOrdering(prompt) {
  const runtimeContextMatch = prompt.match(/<runtime_context>[\s\S]*<\/runtime_context>/);
  if (!runtimeContextMatch) return prompt;

  const runtimeContext = runtimeContextMatch[0];
  const promptWithoutRuntimeContext = prompt.replace(runtimeContext, "").trim();
  const firstNewlineIndex = promptWithoutRuntimeContext.indexOf("\n");

  if (firstNewlineIndex === -1) {
    return `${promptWithoutRuntimeContext}\n\n${runtimeContext}`;
  }

  return `${promptWithoutRuntimeContext.slice(0, firstNewlineIndex)}\n${runtimeContext}\n\n${promptWithoutRuntimeContext.slice(firstNewlineIndex + 1).trim()}`;
}

function maybePadPrompt(prompt) {
  if (!Number.isFinite(DEFAULT_PAD_CHARS) || DEFAULT_PAD_CHARS <= 0) {
    return prompt;
  }

  return `${prompt}\n\n<cache_padding>\n${repeatText(
    "stable-cache-padding",
    DEFAULT_PAD_CHARS,
  )}\n</cache_padding>`;
}

function buildOptimizedOrchestratorPrompt(mode, runtime) {
  const skillsSection = runtime.skills?.length
    ? `\n<skills>
Available user-defined skills:
${runtime.skills
  .map((skill, index) =>
    `${index + 1}. "${skill.title}" (id: ${skill.id})${skill.description ? ` — ${skill.description}` : ""}`,
  )
  .join("\n")}

When you receive a skill reference (skill name + ID) in the user message, call get_skill to load the full instructions, then follow them step-by-step using your available tools.
</skills>`
    : "";

  if (mode === "write") {
    return `You are an orchestrator. Execute actions on integrations or gateways.

TOOLS:
- memory_search: Search for prior context not covered by the user persona above. CORE handles query understanding internally.
- integration_action: Execute an action on a connected service (create, update, delete)
- get_skill: Load a user-defined skill's full instructions by ID. Call this when the request references a skill.
- gateway_*: Offload tasks to connected gateways based on their description

PRIORITY ORDER FOR CONTEXT:
1. User persona above — check here FIRST for preferences, directives, identity, account details (usernames, etc.)
2. memory_search — ONLY if persona doesn't have what you need (prior conversations, specific history, details not in persona)
3. NEVER ask the user for information that's in persona or memory.

If the persona already has the info you need (e.g., github username, preferred channels), skip memory_search and go straight to the action.

RULES:
- ALWAYS search memory first (unless persona already has the info), then execute the action.
- Use memory context to inform how you execute (formatting, recipients, channels, etc.).
- Execute the action. No personality.
- Return result of action (success/failure and details).
- If integration/gateway not connected, say so.
- Match tasks to gateways based on their descriptions.

CRITICAL - FINAL SUMMARY:
When you have completed the action, write a clear, concise summary as your final response.

<runtime_context>
${runtime.persona}
CONNECTED INTEGRATIONS:
${runtime.integrations}

<gateways>
${runtime.gateways}
</gateways>
${skillsSection}
</runtime_context>`;
  }

  return `You are an orchestrator. Gather information based on the intent.

TOOLS:
- memory_search: Search for prior context not covered by the user persona above. CORE handles query understanding internally.
- integration_query: Live data from connected services (emails, calendar, issues, messages)
- web_search: Real-time information from the web (news, current events, documentation, prices, weather, general knowledge). Also use to read/summarize URLs shared by user.
- get_skill: Load a user-defined skill's full instructions by ID. Call this when the request references a skill.
- gateway_*: Offload tasks to connected gateways based on their description (can gather info too)

PRIORITY ORDER FOR CONTEXT:
1. User persona above — check here FIRST for preferences, directives, identity, account details (usernames, etc.)
2. memory_search — if persona doesn't have what you need (prior conversations, specific history, details not in persona)
3. integration_query / web_search — for live data or real-time info
4. NEVER ask the user for information that's in persona or memory.

YOUR JOB:
1. FIRST: Check the user persona above for relevant preferences, directives, and identity info.
2. THEN: If you need prior context not in persona, call memory_search.
3. THEN: Based on the intent AND context, gather from the right source.

RULES:
- Check user persona FIRST.
- Call memory_search for anything not in persona.
- After getting context, proceed with other tools as needed.
- Call multiple tools in parallel when data could be in multiple places.
- No personality. Return raw facts.

<runtime_context>
${runtime.persona}
CONNECTED INTEGRATIONS:
${runtime.integrations}

<gateways>
${runtime.gateways}
</gateways>
${skillsSection}
</runtime_context>`;
}

function buildOptimizedIntegrationExplorerPrompt(mode, runtime) {
  const modeInstructions =
    mode === "write"
      ? "You can CREATE, UPDATE, or DELETE. Execute the requested action."
      : "READ ONLY. Never create, update, or delete.";

  return `You are an Integration Explorer. ${mode === "write" ? "Execute actions on" : "Query"} ONE specific integration.

TOOLS:
- get_integration_actions: Find available actions for a service (returns inputSchema)
- execute_integration_action: Execute an action with parameters matching inputSchema

EXECUTION:
1. Identify which ONE integration matches the request
2. Get actions for that integration
3. Execute the action with parameters matching the schema exactly
4. If you need more detail, fetch it in a second step
5. Return the result

RULES:
- ${modeInstructions}
- Facts only. No personality.
- Target exactly ONE integration per request.

<runtime_context>
NOW: ${runtime.currentDateTime}
TODAY: ${runtime.today}
YESTERDAY: ${runtime.yesterday}

CONNECTED INTEGRATIONS:
${runtime.integrations}
</runtime_context>`;
}

function buildOptimizedWebExplorerPrompt(runtime) {
  return `You are a Web Explorer. Search the web OR read specific URLs.

TOOLS:
- web_search: Search the web for information. Returns titles, URLs, and snippets.
- get_page_contents: Get full content from specific URLs. Use for URLs shared by user or when you need more detail from search results.

RULES:
- Facts only. No personality.
- Cite sources when relevant (include URLs).
- If search returns nothing useful, say so.

<runtime_context>
TODAY: ${runtime.today}
</runtime_context>`;
}

function chatBody(model, messages) {
  const body = {
    model,
    messages,
    temperature: 0,
    max_completion_tokens: 16,
  };

  if (USE_CACHE_HINTS) {
    body.prompt_cache_key = `core-probe:${DEFAULT_MODEL}:exact-prefix`;
    body.prompt_cache_retention = "24h";
  }

  return body;
}

function responsesBody(model, messages) {
  const body = {
    model,
    input: messages.map((message) => ({
      role: message.role,
      content: [{ type: "input_text", text: message.content }],
    })),
    max_output_tokens: 16,
    reasoning: { effort: "low" },
  };

  if (USE_CACHE_HINTS) {
    body.prompt_cache_key = `core-probe:${DEFAULT_MODEL}:exact-prefix`;
    body.prompt_cache_retention = "24h";
  }

  return body;
}

function loadTemplateBody() {
  if (!TEMPLATE_BODY_PATH) return null;
  const resolvedPath = path.resolve(process.cwd(), TEMPLATE_BODY_PATH);
  return JSON.parse(readFileSync(resolvedPath, "utf8"));
}

function extractUsage(json) {
  const usage = json?.usage || {};
  return {
    promptTokens: usage.prompt_tokens ?? usage.input_tokens,
    completionTokens: usage.completion_tokens ?? usage.output_tokens,
    totalTokens: usage.total_tokens,
    cachedInputTokens:
      usage?.prompt_tokens_details?.cached_tokens ??
      usage?.input_tokens_details?.cached_tokens ??
      0,
  };
}

async function runLiveCall(prompt, scenario, variant) {
  const apiKey = EFFECTIVE_ENV.OPENAI_API_KEY;
  if (!apiKey) {
    return { skipped: "OPENAI_API_KEY missing" };
  }

  const baseUrl = DEFAULT_BASE_URL.replace(/\/+$/, "");
  const messages = [
    { role: "system", content: prompt },
    { role: "user", content: 'Reply with exactly "ok".' },
  ];
  const templateBody = loadTemplateBody();

  const url =
    DEFAULT_API_MODE === "responses"
      ? `${baseUrl}/responses`
      : `${baseUrl}/chat/completions`;

  const body =
    templateBody && DEFAULT_API_MODE === "responses"
      ? {
          ...templateBody,
          stream: false,
          store: false,
          prompt_cache_key:
            FRESH_CACHE_KEY || !templateBody.prompt_cache_key
              ? `core-probe:${DEFAULT_MODEL}:${scenario}:${variant}:${Date.now()}`
              : templateBody.prompt_cache_key,
        }
      : DEFAULT_API_MODE === "responses"
        ? responsesBody(DEFAULT_MODEL, messages)
        : chatBody(DEFAULT_MODEL, messages);

  const response = await fetch(url, {
    method: "POST",
    headers: {
      authorization: `Bearer ${apiKey}`,
      "content-type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    return {
      error: `${response.status} ${await response.text()}`,
      scenario,
      variant,
    };
  }

  const json = await response.json();
  return {
    usage: extractUsage(json),
  };
}

async function runProbePair({ name, userMessage, promptA, promptB }, variant, pairName, live) {
  const basePromptA = maybePadPrompt(
    variant === "optimized" ? promptA : toLegacyOrdering(promptA),
  );
  const basePromptB = maybePadPrompt(
    variant === "optimized" ? promptB : toLegacyOrdering(promptB),
  );
  const chosenA = basePromptA;
  const chosenB = pairName === "exact-repeat" ? basePromptA : basePromptB;

  return {
    scenario: name,
    userMessage,
    variant,
    pairName,
    prompt1: serializePromptMetrics(chosenA),
    prompt2: serializePromptMetrics(chosenB),
    sharedPrefixChars: sharedPrefixChars(chosenA, chosenB),
    liveRun1: live ? await runLiveCall(chosenA, name, variant) : { skipped: "dry-run" },
    liveRun2: live ? await runLiveCall(chosenB, name, variant) : { skipped: "dry-run" },
  };
}

function buildScenarios() {
  return [
    {
      name: "orchestrator-read",
      userMessage: "What did we decide about the auth proxy path in the core project?",
      promptA: buildOptimizedOrchestratorPrompt("read", {
        persona: `${DEFAULT_PERSONA}\n- probe stamp: alpha`,
        integrations: `${DEFAULT_INTEGRATIONS}\n4. **Linear** (Account ID: linear-acct-1) (Identifier: core-team)`,
        gateways: DEFAULT_GATEWAYS,
        skills: [
          { id: "skill-plan-day", title: "Plan My Day", description: "Turn calendar and inbox into an actionable day plan" },
          { id: "skill-review-pr", title: "Review PR", description: "Pull context, inspect diff, summarize risks" },
        ],
      }),
      promptB: buildOptimizedOrchestratorPrompt("read", {
        persona: `${DEFAULT_PERSONA}\n- probe stamp: beta`,
        integrations: `${DEFAULT_INTEGRATIONS}\n4. **Linear** (Account ID: linear-acct-2) (Identifier: core-team)`,
        gateways: DEFAULT_GATEWAYS,
        skills: [
          { id: "skill-plan-day", title: "Plan My Day", description: "Turn calendar and inbox into an actionable day plan" },
          { id: "skill-review-pr", title: "Review PR", description: "Pull context, inspect diff, summarize risks" },
        ],
      }),
    },
    {
      name: "orchestrator-write",
      userMessage: "Fix the auth bug in core repo and create a GitHub issue if needed.",
      promptA: buildOptimizedOrchestratorPrompt("write", {
        persona: `${DEFAULT_PERSONA}\n- probe stamp: alpha`,
        integrations: DEFAULT_INTEGRATIONS,
        gateways: DEFAULT_GATEWAYS,
        skills: [],
      }),
      promptB: buildOptimizedOrchestratorPrompt("write", {
        persona: `${DEFAULT_PERSONA}\n- probe stamp: beta`,
        integrations: DEFAULT_INTEGRATIONS,
        gateways: DEFAULT_GATEWAYS,
        skills: [],
      }),
    },
    {
      name: "integration-explorer-read",
      userMessage: "summarize the PR for auth fix",
      promptA: buildOptimizedIntegrationExplorerPrompt("read", {
        currentDateTime: "2026-04-14 03:15:30",
        today: "2026-04-14",
        yesterday: "2026-04-13",
        integrations: DEFAULT_INTEGRATIONS,
      }),
      promptB: buildOptimizedIntegrationExplorerPrompt("read", {
        currentDateTime: "2026-04-14 03:16:05",
        today: "2026-04-14",
        yesterday: "2026-04-13",
        integrations: DEFAULT_INTEGRATIONS,
      }),
    },
    {
      name: "web-explorer-read",
      userMessage: "latest news about AI regulation",
      promptA: buildOptimizedWebExplorerPrompt({
        today: "2026-04-14 (America/Los_Angeles)",
      }),
      promptB: buildOptimizedWebExplorerPrompt({
        today: "2026-04-15 (America/Los_Angeles)",
      }),
    },
  ];
}

async function main() {
  const requestedVariant = process.argv
    .find((arg) => arg.startsWith("--variant="))
    ?.split("=")[1];
  const requestedPair = process.argv
    .find((arg) => arg.startsWith("--pair="))
    ?.split("=")[1];
  const requestedScenario = process.argv
    .find((arg) => arg.startsWith("--scenario="))
    ?.split("=")[1];
  const live = process.argv.includes("--live");
  const variants = requestedVariant ? [requestedVariant] : ["legacy", "optimized"];
  const pairs = requestedPair ? [requestedPair] : ["exact-repeat", "volatile-repeat"];
  const scenarios = buildScenarios().filter((scenario) =>
    requestedScenario ? scenario.name === requestedScenario : true,
  );

  if (scenarios.length === 0) {
    throw new Error(`No matching scenario for ${requestedScenario}`);
  }

  if (!["legacy", "optimized"].includes(variants[0])) {
    throw new Error(`Unsupported variant ${requestedVariant}`);
  }

  if (!["exact-repeat", "volatile-repeat"].includes(pairs[0])) {
    throw new Error(`Unsupported pair ${requestedPair}`);
  }

  const results = [];
  for (const scenario of scenarios) {
    for (const variant of variants) {
      for (const pairName of pairs) {
        results.push(await runProbePair(scenario, variant, pairName, live));
      }
    }
  }

  console.log(
    JSON.stringify(
      {
        live,
        apiMode: DEFAULT_API_MODE,
        baseUrl: DEFAULT_BASE_URL,
        model: DEFAULT_MODEL,
        padStaticChars: DEFAULT_PAD_CHARS,
        withCacheHints: USE_CACHE_HINTS,
        templateBodyPath: TEMPLATE_BODY_PATH || null,
        freshCacheKey: FRESH_CACHE_KEY,
        envFiles: EFFECTIVE_ENV.__files,
        scenarios: scenarios.map((scenario) => scenario.name),
        results,
      },
      null,
      2,
    ),
  );
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack || error.message : String(error));
  process.exitCode = 1;
});
