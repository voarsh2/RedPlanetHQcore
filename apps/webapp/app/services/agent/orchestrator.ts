import {
  streamText,
  type LanguageModel,
  stepCountIs,
  tool,
} from "ai";
import { z } from "zod";

import { runIntegrationExplorer, runWebExplorer } from "./explorers";
import { searchMemoryWithAgent } from "./memory";
import { logger } from "../logger.service";
import { IntegrationLoader } from "~/utils/mcp/integration-loader";
import { makeModelCall } from "~/lib/model.server";
import { getGatewayAgents, runGatewayExplorer } from "./gateway";
import { type SkillRef } from "./types";
import { prisma } from "~/db.server";
import {
  compactNestedResult,
  type ToolProgressSink,
  withProgressHeartbeat,
} from "./tool-progress";

export type OrchestratorMode = "read" | "write";

export const buildOrchestratorPrompt = (
  integrations: string,
  mode: OrchestratorMode,
  gateways: string,
  userPersona?: string,
  skills?: SkillRef[],
) => {
  const personaSection = userPersona
    ? `\nUSER PERSONA (identity, preferences, directives - use this FIRST before searching memory):\n${userPersona}\n`
    : "";

  const skillsSection =
    skills && skills.length > 0
      ? `\n<skills>
Available user-defined skills:
${skills.map((s, i) => {
        const meta = s.metadata as Record<string, unknown> | null;
        const desc = meta?.shortDescription as string | undefined;
        return `${i + 1}. "${s.title}" (id: ${s.id})${desc ? ` — ${desc}` : ""}`;
      }).join("\n")}

When you receive a skill reference (skill name + ID) in the user message, call get_skill to load the full instructions, then follow them step-by-step using your available tools.
</skills>\n`
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

EXAMPLES:

Action: "send a slack message to #general saying standup in 5"
Step 1: memory_search("user's preferences for slack messages - preferred channels, formatting, any standing directives about team communication")
Step 2: integration_action({ integration: "slack", action: "send message to #general: standup in 5" })

Action: "create a github issue for auth bug in core repo"
Step 1: memory_search("user's preferences for github issues - preferred repos, labels, templates, any directives about issue creation")
Step 2: integration_action({ integration: "github", action: "create issue titled auth bug in core repo" })

Action: "fix the auth bug in core repo" (gateway description: "personal coding tasks")
Execute: gateway_harshith_mac({ intent: "fix the auth bug in core repo" })

RULES:
- ALWAYS search memory first (unless persona already has the info), then execute the action.
- Use memory context to inform how you execute (formatting, recipients, channels, etc.).
- Execute the action. No personality.
- Return result of action (success/failure and details).
- If integration/gateway not connected, say so.
- Match tasks to gateways based on their descriptions.

CRITICAL - FINAL SUMMARY:
When you have completed the action, write a clear, concise summary as your final response.
This summary will be returned to the parent agent, so include:
- What action was performed
- The result (success/failure)
- Any relevant details (IDs, URLs, error messages)

Example final summary: "Created GitHub issue #123 'Fix auth bug' in core repo. URL: https://github.com/org/core/issues/123"

<runtime_context>
${personaSection ? `${personaSection.trim()}\n` : ""}CONNECTED INTEGRATIONS:
${integrations}

<gateways>
${gateways || "No gateways connected"}
</gateways>
${skillsSection ? skillsSection.trim() : ""}
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
3. THEN: Based on the intent AND context, gather from the right source:
   - Integrations: live/current data from external services (user's emails, calendar, issues)
   - Web: real-time info not in memory or integrations (news, weather, docs, how-tos, current events, general questions)
   - Multiple: when you need info from several sources

CRITICAL FOR memory_search:
- Describe your INTENT - what you need from memory and why.
- Always include: preferences, directives, and prior context related to the request.
- Write it like asking a colleague to find something.
- CORE has agentic search that understands natural language.

BAD (keyword soup - will fail):
- "rerank evaluation metrics NDCG MRR pairwise pointwise"
- "Manoj Sol rerank evaluation dataset methodology"

GOOD (clear intent):
- "Find user preferences, directives, and past discussions about rerank evaluation - what approach was decided, any metrics discussed, next steps"
- "What has user said about their morning routine preferences and productivity habits"
- "User's preferences and previous conversations about the deployment plan and any blockers mentioned"

EXAMPLES:

Intent: "What did we discuss about the marketing strategy"
Step 1: memory_search("User preferences, directives, and past discussions about marketing strategy - decisions made, timeline, who's involved")

Intent: "Show me my upcoming meetings this week"
Step 1: memory_search("User's preferences and directives about calendar, meetings, and scheduling")
Step 2: integration_query: google-calendar (live data)

Intent: "Status of the deployment - what we planned vs current blockers"
Step 1: memory_search("User preferences, directives, and previous discussions about deployment planning and decisions")
Step 2: integration_query: github/linear

Intent: "What's the weather in San Francisco"
Step 1: memory_search("User's location preferences, directives about weather updates")
Step 2: web_search (real-time data)

Intent: "Latest news about AI regulation"
Step 1: memory_search("User's interests and directives about AI news and regulation topics")
Step 2: web_search (current events)

Intent: "What newsletters came in today" / "my GitHub notifications"
Step 1: memory_search("User's preferences for email filtering, notification priorities, and directives about newsletters")
Step 2: integration_query: gmail (user's personal inbox)

Intent: "summarize this: https://example.com/article"
→ web_search (reads the URL content) — no memory search needed for pure URL fetching

BE PROACTIVE:
- If a specific query returns empty, try a broader one to validate data exists.
- If memory returns nothing on a specific topic, try related topics before reporting empty.
- If integration returns empty, confirm the resource exists (repo, channel, calendar) before saying "nothing found".

RULES:
- Check user persona FIRST — it has identity, preferences, directives.
- Call memory_search for anything not in persona (prior conversations, specific history).
- NEVER ask the user for info that's already in persona or memory.
- After getting context, proceed with other tools as needed.
- Call multiple tools in parallel when data could be in multiple places.
- No personality. Return raw facts.

<runtime_context>
${personaSection ? `${personaSection.trim()}\n` : ""}CONNECTED INTEGRATIONS:
${integrations}

<gateways>
${gateways || "No gateways connected"}
</gateways>
${skillsSection ? skillsSection.trim() : ""}
</runtime_context>`;
};

export interface OrchestratorResult {
  stream: ReturnType<typeof streamText>;
  startTime: number;
  progressId?: string;
}

export async function runOrchestrator(
  userId: string,
  workspaceId: string,
  userMessage: string,
  mode: OrchestratorMode = "read",
  timezone: string = "UTC",
  source: string,
  abortSignal?: AbortSignal,
  userPersona?: string,
  skills?: SkillRef[],
  progressSink?: ToolProgressSink,
  progressParentId?: string,
): Promise<OrchestratorResult> {
  const startTime = Date.now();
  const orchestratorProgressId = `orchestrator:${crypto.randomUUID()}`;

  // Get user's connected integrations
  const connectedIntegrations =
    await IntegrationLoader.getConnectedIntegrationAccounts(
      userId,
      workspaceId,
    );

  const integrationsList = connectedIntegrations
    .map(
      (int, index) =>
        `${index + 1}. **${int.integrationDefinition.name}** (Account ID: ${int.id}) (Identifier: ${int.accountId})`,
    )
    .join("\n");

  // Get connected gateways
  const gateways = await getGatewayAgents(workspaceId);
  const gatewaysList = gateways
    .map(
      (gw, index) =>
        `${index + 1}. **${gw.name}** (tool: gateway_${gw.name.toLowerCase().replace(/[^a-z0-9]/g, "_")}): ${gw.description}`,
    )
    .join("\n");

  logger.info(
    `Orchestrator: Loaded ${connectedIntegrations.length} integrations, ${gateways.length} gateways, mode: ${mode}`,
  );
  await progressSink?.({
    id: orchestratorProgressId,
    parentId: progressParentId,
    level: 1,
    label: "Orchestrator",
    status: "running",
    detail: mode === "read" ? "Choosing context sources" : "Choosing action path",
    elapsedMs: 0,
  });

  // Build tools based on mode
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const tools: Record<string, any> = {};

  // memory_search is available in both read and write modes
  tools.memory_search = tool({
    description:
      "Search user preferences, directives, past conversations, and stored knowledge. ALWAYS call this FIRST before any other tool.",
    inputSchema: z.object({
      query: z
        .string()
        .describe(
          "What to search for - include preferences, directives, and prior context related to the request",
        ),
    }),
    execute: async ({ query }) => {
      const toolRunId = crypto.randomUUID();
      const progressId = `memory-search:${toolRunId}`;
      const startedAt = Date.now();
      logger.info(`Orchestrator: memory search - ${query}`);
      await progressSink?.({
        id: progressId,
        parentId: orchestratorProgressId,
        level: 2,
        label: "Memory explorer",
        status: "running",
        detail: query,
        elapsedMs: 0,
      });
      try {
        const result = await withProgressHeartbeat(
          searchMemoryWithAgent(
            query,
            userId,
            workspaceId,
            source,
            { structured: false },
          ),
          {
            sink: progressSink,
            event: {
              id: progressId,
              parentId: orchestratorProgressId,
              level: 2,
              label: "Memory explorer",
              detail: "Memory search still running",
            },
            startedAt,
            intervalMs: 10_000,
          },
        );
        await progressSink?.({
          id: progressId,
          parentId: orchestratorProgressId,
          level: 2,
          label: "Memory explorer",
          status: "completed",
          detail: result ? "Memory search completed" : "No memory results found",
          elapsedMs: Date.now() - startedAt,
        });
        return result || "nothing found";
      } catch (error: any) {
        logger.warn("Memory search failed", error);
        await progressSink?.({
          id: progressId,
          parentId: orchestratorProgressId,
          level: 2,
          label: "Memory explorer",
          status: "failed",
          detail: error instanceof Error ? error.message : String(error),
          elapsedMs: Date.now() - startedAt,
        });
        return "nothing found";
      }
    },
  });

  // get_skill tool - available in both modes when skills exist
  if (skills && skills.length > 0) {
    tools.get_skill = tool({
      description:
        "Load a user-defined skill's full instructions by ID. Call this when the request references a skill, then follow the instructions step-by-step.",
      inputSchema: z.object({
        skill_id: z
          .string()
          .describe("The skill ID to load"),
      }),
      execute: async ({ skill_id }) => {
        logger.info(`Orchestrator: loading skill ${skill_id}`);
        try {
          const skill = await prisma.document.findFirst({
            where: {
              id: skill_id,
              workspaceId,
              type: "skill",
              deleted: null,
            },
            select: { id: true, title: true, content: true },
          });
          if (!skill) return "Skill not found";
          return `## Skill: ${skill.title}\n\n${skill.content}`;
        } catch (error: any) {
          logger.warn("Failed to load skill", error);
          return "Failed to load skill";
        }
      },
    });
  }

  if (mode === "read") {
    tools.integration_query = tool({
      description: "Query a connected integration for current data",
      inputSchema: z.object({
        integration: z
          .string()
          .describe(
            "Which integration to query (e.g., github, slack, notion, google-calendar, gmail)",
          ),
        query: z.string().describe("What data to get"),
      }),
      execute: async function* ({ integration, query }, { abortSignal }) {
        const toolRunId = crypto.randomUUID();
        const progressId = `integration-query:${toolRunId}`;
        const startedAt = Date.now();
        logger.info(
          `Orchestrator: integration query - ${integration}: ${query}`,
        );
        await progressSink?.({
          id: progressId,
          parentId: orchestratorProgressId,
          level: 2,
          label: `${integration} explorer`,
          status: "running",
          detail: query,
          elapsedMs: 0,
        });

        const { stream, hasIntegrations } = await runIntegrationExplorer(
          `${query} from ${integration}`,
          integrationsList,
          "read",
          timezone,
          source,
          userId,
          abortSignal,
          progressSink,
          progressId,
        );
        logger.info("Orchestrator: integration_query nested stream created", {
          toolRunId,
          elapsedMs: Date.now() - startedAt,
          integration,
          queryChars: query.length,
          hasIntegrations,
        });

        if (!hasIntegrations) {
          await progressSink?.({
            id: progressId,
            parentId: orchestratorProgressId,
            level: 2,
            label: `${integration} explorer`,
            status: "failed",
            detail: "No integrations connected",
            elapsedMs: Date.now() - startedAt,
          });
          yield {
            parts: [{ type: "text", text: "No integrations connected" }],
          };
          logger.info("Orchestrator: integration_query skipped no integrations", {
            toolRunId,
            elapsedMs: Date.now() - startedAt,
            integration,
          });
          return;
        }

        try {
          const finalText = await stream.text;
          await progressSink?.({
            id: progressId,
            parentId: orchestratorProgressId,
            level: 2,
            label: `${integration} explorer`,
            status: "completed",
            detail: "Integration query complete",
            elapsedMs: Date.now() - startedAt,
          });
          yield compactNestedResult(
            finalText,
            `Integration query for ${integration} completed, but no final summary was produced.`,
          );
          logger.info("Orchestrator: integration_query nested stream closed", {
            toolRunId,
            elapsedMs: Date.now() - startedAt,
            integration,
            finalTextChars: finalText?.length ?? 0,
          });
        } catch (error) {
          await progressSink?.({
            id: progressId,
            parentId: orchestratorProgressId,
            level: 2,
            label: `${integration} explorer`,
            status: "failed",
            detail: error instanceof Error ? error.message : String(error),
            elapsedMs: Date.now() - startedAt,
          });
          logger.error("Orchestrator: integration_query nested stream failed", {
            toolRunId,
            elapsedMs: Date.now() - startedAt,
            integration,
            error,
          });
          throw error;
        }
      },
    });

    tools.web_search = tool({
      description:
        "Search the web for real-time information: news, current events, documentation, prices, weather, general knowledge. Use when info is not in memory or integrations.",
      inputSchema: z.object({
        query: z
          .string()
          .describe("What to search for - be specific and clear"),
      }),
      execute: async ({ query }) => {
        const toolRunId = crypto.randomUUID();
        const startedAt = Date.now();
        logger.info(`Orchestrator: web search - ${query}`);
        await progressSink?.({
          id: `web-search:${toolRunId}`,
          parentId: orchestratorProgressId,
          level: 2,
          label: "Web search",
          status: "running",
          detail: query,
          elapsedMs: 0,
        });
        const result = await runWebExplorer(query, timezone);
        await progressSink?.({
          id: `web-search:${toolRunId}`,
          parentId: orchestratorProgressId,
          level: 2,
          label: "Web search",
          status: result.success ? "completed" : "failed",
          detail: result.success ? "Web search completed" : "Web search unavailable",
          elapsedMs: Date.now() - startedAt,
        });
        return result.success ? result.data : "web search unavailable";
      },
    });
  } else {
    // Write mode - action tool
    tools.integration_action = tool({
      description:
        "Execute an action on a connected integration (create, send, update, delete)",
      inputSchema: z.object({
        integration: z
          .string()
          .describe(
            "Which integration to use (e.g., github, slack, notion, google-calendar, gmail)",
          ),
        action: z.string().describe("What action to perform, be specific"),
      }),
      execute: async function* ({ integration, action }, { abortSignal }) {
        const toolRunId = crypto.randomUUID();
        const progressId = `integration-action:${toolRunId}`;
        const startedAt = Date.now();
        logger.info(
          `Orchestrator: integration action - ${integration}: ${action}`,
        );
        await progressSink?.({
          id: progressId,
          parentId: orchestratorProgressId,
          level: 2,
          label: `${integration} action`,
          status: "running",
          detail: action,
          elapsedMs: 0,
        });

        const { stream, hasIntegrations } = await runIntegrationExplorer(
          `${action} on ${integration}`,
          integrationsList,
          "write",
          timezone,
          source,
          userId,
          abortSignal,
          progressSink,
          progressId,
        );
        logger.info("Orchestrator: integration_action nested stream created", {
          toolRunId,
          elapsedMs: Date.now() - startedAt,
          integration,
          actionChars: action.length,
          hasIntegrations,
        });

        if (!hasIntegrations) {
          await progressSink?.({
            id: progressId,
            parentId: orchestratorProgressId,
            level: 2,
            label: `${integration} action`,
            status: "failed",
            detail: "No integrations connected",
            elapsedMs: Date.now() - startedAt,
          });
          yield {
            parts: [{ type: "text", text: "No integrations connected" }],
          };
          logger.info("Orchestrator: integration_action skipped no integrations", {
            toolRunId,
            elapsedMs: Date.now() - startedAt,
            integration,
          });
          return;
        }

        try {
          const finalText = await stream.text;
          await progressSink?.({
            id: progressId,
            parentId: orchestratorProgressId,
            level: 2,
            label: `${integration} action`,
            status: "completed",
            detail: "Integration action complete",
            elapsedMs: Date.now() - startedAt,
          });
          yield compactNestedResult(
            finalText,
            `Integration action for ${integration} completed, but no final summary was produced.`,
          );
          logger.info("Orchestrator: integration_action nested stream closed", {
            toolRunId,
            elapsedMs: Date.now() - startedAt,
            integration,
            finalTextChars: finalText?.length ?? 0,
          });
        } catch (error) {
          await progressSink?.({
            id: progressId,
            parentId: orchestratorProgressId,
            level: 2,
            label: `${integration} action`,
            status: "failed",
            detail: error instanceof Error ? error.message : String(error),
            elapsedMs: Date.now() - startedAt,
          });
          logger.error("Orchestrator: integration_action nested stream failed", {
            toolRunId,
            elapsedMs: Date.now() - startedAt,
            integration,
            error,
          });
          throw error;
        }
      },
    });
  }

  // Add gateway tools for both modes
  for (const gateway of gateways) {
    if (gateway.status !== "CONNECTED") continue;

    const toolName = `gateway_${gateway.name.toLowerCase().replace(/[^a-z0-9]/g, "_")}`;

    tools[toolName] = tool({
      description: `**${gateway.name}** - ${gateway.description}`,
      inputSchema: z.object({
        intent: z
          .string()
          .describe(
            "Describe what you want to accomplish. Be specific about the task.",
          ),
      }),
      execute: async function* ({ intent }, { abortSignal }) {
        const toolRunId = crypto.randomUUID();
        const progressId = `gateway:${toolRunId}`;
        const startedAt = Date.now();
        logger.info(`Orchestrator: Gateway ${gateway.name} - ${intent}`);
        await progressSink?.({
          id: progressId,
          parentId: orchestratorProgressId,
          level: 2,
          label: `Gateway: ${gateway.name}`,
          status: "running",
          detail: intent,
          elapsedMs: 0,
        });

        const { stream, gatewayConnected } = await runGatewayExplorer(
          gateway.id,
          intent,
          abortSignal,
          progressSink,
          progressId,
        );
        logger.info("Orchestrator: gateway nested stream created", {
          toolRunId,
          elapsedMs: Date.now() - startedAt,
          gatewayName: gateway.name,
          intentChars: intent.length,
          gatewayConnected,
          hasStream: Boolean(stream),
        });

        if (!gatewayConnected || !stream) {
          await progressSink?.({
            id: progressId,
            parentId: orchestratorProgressId,
            level: 2,
            label: `Gateway: ${gateway.name}`,
            status: "failed",
            detail: "Gateway is not connected",
            elapsedMs: Date.now() - startedAt,
          });
          yield {
            parts: [
              {
                type: "text",
                text: `Gateway "${gateway.name}" is not connected.`,
              },
            ],
          };
          logger.info("Orchestrator: gateway skipped not connected", {
            toolRunId,
            elapsedMs: Date.now() - startedAt,
            gatewayName: gateway.name,
          });
          return;
        }

        try {
          const finalText = await stream.text;
          await progressSink?.({
            id: progressId,
            parentId: orchestratorProgressId,
            level: 2,
            label: `Gateway: ${gateway.name}`,
            status: "completed",
            detail: "Gateway task complete",
            elapsedMs: Date.now() - startedAt,
          });
          yield compactNestedResult(
            finalText,
            `Gateway "${gateway.name}" completed, but no final summary was produced.`,
          );
          logger.info("Orchestrator: gateway nested stream closed", {
            toolRunId,
            elapsedMs: Date.now() - startedAt,
            gatewayName: gateway.name,
            finalTextChars: finalText?.length ?? 0,
          });
        } catch (error) {
          await progressSink?.({
            id: progressId,
            parentId: orchestratorProgressId,
            level: 2,
            label: `Gateway: ${gateway.name}`,
            status: "failed",
            detail: error instanceof Error ? error.message : String(error),
            elapsedMs: Date.now() - startedAt,
          });
          logger.error("Orchestrator: gateway nested stream failed", {
            toolRunId,
            elapsedMs: Date.now() - startedAt,
            gatewayName: gateway.name,
            error,
          });
          throw error;
        }
      },
    });
  }

  const stream = await makeModelCall(
    true,
    [
      {
        role: "system",
        content: buildOrchestratorPrompt(
          integrationsList,
          mode,
          gatewaysList,
          userPersona,
          skills,
        ),
      },
      { role: "user", content: userMessage },
    ],
    () => {},
    {
      tools,
      stopWhen: stepCountIs(10),
      abortSignal,
    },
    "high",
    `core-orchestrator-${mode}`,
    undefined,
    { callSite: `core.orchestrator.${mode}` },
  );

  logger.info(`Orchestrator: Starting stream for mode ${mode}`);

  return {
    stream,
    startTime,
    progressId: progressSink ? orchestratorProgressId : undefined,
  };
}
