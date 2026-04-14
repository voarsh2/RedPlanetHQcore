#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";

function parseArgs(argv) {
  const args = {
    configPath: path.join(os.homedir(), ".codex", "config.toml"),
    authPath: path.join(os.homedir(), ".codex", "auth.json"),
    templateBodyPath: "",
    model: "gpt-5.4",
    serviceTier: "priority",
    promptCacheKey: "codex-cache-test-1",
    padBlocks: 400,
    callRetries: 20,
    pairRetries: 1,
    delayMs: 2000,
    timeoutMs: 120000,
    stream: false,
    includeReasoningEncrypted: true,
    addClientMetadata: true,
    addChatgptAccountHeader: true,
    addOriginatorHeader: true,
    originator: "codex_vscode",
    installationId: "",
    outputPath: "",
  };

  for (const arg of argv) {
    if (!arg.startsWith("--")) continue;
    const [rawKey, rawValue = "true"] = arg.slice(2).split("=");
    const key = rawKey.replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    const value = rawValue;
    if (key in args) {
      if (typeof args[key] === "number") {
        args[key] = Number(value);
      } else if (typeof args[key] === "boolean") {
        args[key] = value !== "false";
      } else {
        args[key] = value;
      }
    }
  }

  return args;
}

function readCodexConfig(configPath) {
  const text = fs.readFileSync(configPath, "utf8");
  const baseUrl = text.match(/^\s*base_url\s*=\s*"([^"]+)"/m)?.[1];
  const wireApi = text.match(/^\s*wire_api\s*=\s*"([^"]+)"/m)?.[1];
  const token = text.match(/^\s*experimental_bearer_token\s*=\s*"([^"]+)"/m)?.[1];

  if (!baseUrl || !token) {
    throw new Error(`Could not read Codex base_url/token from ${configPath}`);
  }

  return { baseUrl, wireApi: wireApi ?? "responses", token };
}

function readCodexAuth(authPath) {
  const text = fs.readFileSync(authPath, "utf8");
  const auth = JSON.parse(text);
  return {
    accountId: auth?.tokens?.account_id ?? "",
  };
}

function buildPad(blocks) {
  return Array.from(
    { length: blocks },
    (_, i) =>
      `STATIC_BLOCK_${String(i).padStart(4, "0")} Lorem ipsum cache probe line for codex route.`
  ).join(" ");
}

function buildSyntheticBody({
  model,
  serviceTier,
  promptCacheKey,
  padBlocks,
  stream,
  includeReasoningEncrypted,
  addClientMetadata,
  installationId,
}) {
  const pad = buildPad(padBlocks);
  const body = {
    model,
    instructions: "You are Codex, a coding agent based on GPT-5.",
    input: [
      {
        type: "message",
        role: "user",
        content: [
          {
            type: "input_text",
            text: `Reply with the single word ok.\n\n${pad}`,
          },
        ],
      },
    ],
    tools: [],
    tool_choice: "auto",
    parallel_tool_calls: true,
    reasoning: { effort: "low" },
    store: false,
    stream,
    service_tier: serviceTier,
    prompt_cache_key: promptCacheKey,
    text: { verbosity: "low" },
  };

  if (includeReasoningEncrypted) {
    body.include = ["reasoning.encrypted_content"];
  }

  if (addClientMetadata && installationId) {
    body.client_metadata = {
      "x-codex-installation-id": installationId,
    };
  }

  return body;
}

function updateTemplateBody(template, args) {
  const body = structuredClone(template);
  body.model = args.model || body.model;
  body.service_tier = args.serviceTier;
  body.prompt_cache_key = args.promptCacheKey;
  body.stream = args.stream;

  if (args.includeReasoningEncrypted) {
    body.include = ["reasoning.encrypted_content"];
  }

  if (args.addClientMetadata && args.installationId) {
    body.client_metadata = {
      ...(body.client_metadata ?? {}),
      "x-codex-installation-id": args.installationId,
    };
  }

  return body;
}

function parseSse(text) {
  const events = [];
  for (const chunk of text.split(/\n\n+/)) {
    const lines = chunk.split(/\r?\n/).filter(Boolean);
    const event = lines.find((line) => line.startsWith("event: "))?.slice(7);
    const dataLines = lines
      .filter((line) => line.startsWith("data: "))
      .map((line) => line.slice(6));

    if (!event) continue;

    const dataText = dataLines.join("\n");
    let data = dataText;
    try {
      data = JSON.parse(dataText);
    } catch {
      // Keep raw text when it is not JSON.
    }
    events.push({ event, data });
  }
  return events;
}

function summarizeResponse(events) {
  const failed = events.find((entry) => entry.event === "response.failed");
  if (failed) {
    const response = failed.data?.response ?? {};
    return {
      status: "failed",
      id: response.id ?? null,
      errorCode: response.error?.code ?? null,
      errorMessage: response.error?.message ?? null,
      usage: null,
      outputText: null,
    };
  }

  const completed = events.find((entry) => entry.event === "response.completed");
  if (!completed) {
    return {
      status: "missing_completed",
      id: null,
      errorCode: null,
      errorMessage: "No response.completed event found",
      usage: null,
      outputText: null,
    };
  }

  const response = completed.data?.response ?? {};
  const outputText =
    response.output
      ?.flatMap((item) => item.content ?? [])
      .find((item) => item.type === "output_text")?.text ?? null;

  return {
    status: response.status ?? "completed",
    id: response.id ?? null,
    errorCode: null,
    errorMessage: null,
    usage: response.usage ?? null,
    outputText,
  };
}

async function runRequest({
  url,
  token,
  body,
  timeoutMs,
  accountId,
  addChatgptAccountHeader,
  addOriginatorHeader,
  originator,
}) {
  const headers = {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
  };

  if (addChatgptAccountHeader && accountId) {
    headers["ChatGPT-Account-Id"] = accountId;
  }

  if (addOriginatorHeader && originator) {
    headers.originator = originator;
  }

  const response = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });

  const text = await response.text();
  const events = parseSse(text);
  const summary = summarizeResponse(events);

  return {
    httpStatus: response.status,
    text,
    events,
    summary,
  };
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function runLogicalCall({
  label,
  url,
  token,
  body,
  callRetries,
  delayMs,
  timeoutMs,
  accountId,
  addChatgptAccountHeader,
  addOriginatorHeader,
  originator,
}) {
  const attempts = [];

  for (let i = 1; i <= callRetries; i += 1) {
    const result = await runRequest({
      url,
      token,
      body,
      timeoutMs,
      accountId,
      addChatgptAccountHeader,
      addOriginatorHeader,
      originator,
    });
    attempts.push({
      attempt: i,
      httpStatus: result.httpStatus,
      status: result.summary.status,
      errorCode: result.summary.errorCode,
      errorMessage: result.summary.errorMessage,
      usage: result.summary.usage,
      outputText: result.summary.outputText,
    });

    if (result.summary.status === "completed") {
      return {
        label,
        success: true,
        attempts,
        final: attempts.at(-1),
        rawText: result.text,
      };
    }

    await sleep(delayMs);
  }

  return {
    label,
    success: false,
    attempts,
    final: attempts.at(-1) ?? null,
    rawText: "",
  };
}

function formatUsage(usage) {
  if (!usage) return null;
  return {
    input_tokens: usage.input_tokens ?? null,
    cached_tokens: usage.input_tokens_details?.cached_tokens ?? null,
    output_tokens: usage.output_tokens ?? null,
    total_tokens: usage.total_tokens ?? null,
    reasoning_tokens: usage.output_tokens_details?.reasoning_tokens ?? null,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const config = readCodexConfig(args.configPath);
  const auth = fs.existsSync(args.authPath) ? readCodexAuth(args.authPath) : { accountId: "" };

  if (config.wireApi !== "responses") {
    throw new Error(`Expected wire_api="responses", got "${config.wireApi}"`);
  }

  const url = `${config.baseUrl.replace(/\/+$/, "")}/responses`;
  const body = args.templateBodyPath
    ? updateTemplateBody(
        JSON.parse(fs.readFileSync(args.templateBodyPath, "utf8")),
        args
      )
    : buildSyntheticBody(args);

  const pairSummaries = [];

  for (let pairIndex = 1; pairIndex <= args.pairRetries; pairIndex += 1) {
    const first = await runLogicalCall({
      label: "first",
      url,
      token: config.token,
      body,
      callRetries: args.callRetries,
      delayMs: args.delayMs,
      timeoutMs: args.timeoutMs,
      accountId: auth.accountId,
      addChatgptAccountHeader: args.addChatgptAccountHeader,
      addOriginatorHeader: args.addOriginatorHeader,
      originator: args.originator,
    });

    if (!first.success) {
      pairSummaries.push({ pairIndex, first, second: null });
      continue;
    }

    await sleep(args.delayMs);

    const second = await runLogicalCall({
      label: "second",
      url,
      token: config.token,
      body,
      callRetries: args.callRetries,
      delayMs: args.delayMs,
      timeoutMs: args.timeoutMs,
      accountId: auth.accountId,
      addChatgptAccountHeader: args.addChatgptAccountHeader,
      addOriginatorHeader: args.addOriginatorHeader,
      originator: args.originator,
    });

    pairSummaries.push({ pairIndex, first, second });

    if (second.success) break;
  }

  const result = {
    config: {
      baseUrl: config.baseUrl,
      wireApi: config.wireApi,
      model: args.model,
      serviceTier: args.serviceTier,
      promptCacheKey: args.promptCacheKey,
      padBlocks: args.padBlocks,
      templateBodyPath: args.templateBodyPath,
      callRetries: args.callRetries,
      pairRetries: args.pairRetries,
      delayMs: args.delayMs,
      addChatgptAccountHeader: args.addChatgptAccountHeader,
      addOriginatorHeader: args.addOriginatorHeader,
      originator: args.originator,
      accountIdPresent: Boolean(auth.accountId),
    },
    pairs: pairSummaries.map(({ pairIndex, first, second }) => ({
      pairIndex,
      first: first
        ? {
            success: first.success,
            final: first.final
              ? {
                  ...first.final,
                  usage: formatUsage(first.final.usage),
                }
              : null,
            attempts: first.attempts.map((attempt) => ({
              ...attempt,
              usage: formatUsage(attempt.usage),
            })),
          }
        : null,
      second: second
        ? {
            success: second.success,
            final: second.final
              ? {
                  ...second.final,
                  usage: formatUsage(second.final.usage),
                }
              : null,
            attempts: second.attempts.map((attempt) => ({
              ...attempt,
              usage: formatUsage(attempt.usage),
            })),
          }
        : null,
    })),
  };

  const json = JSON.stringify(result, null, 2);
  if (args.outputPath) {
    fs.writeFileSync(args.outputPath, json);
  }
  console.log(json);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack : String(error));
  process.exitCode = 1;
});
