import { type Tool, tool } from "ai";
import { z } from "zod";

import { runOrchestrator } from "./orchestrator";
import { type SkillRef } from "./types";

import { logger } from "../logger.service";
import { getReminderTools } from "./tools/reminder-tools";
import {
  compactNestedResult,
  createToolProgressRelay,
} from "./tool-progress";

export const createTools = async (
  userId: string,
  workspaceId: string,
  timezone: string,
  source: string,
  readOnly: boolean = false,
  persona?: string,
  skills?: SkillRef[],
  /** Optional callback for channels to send intermediate messages (acks) */
  onMessage?: (message: string) => Promise<void>,
) => {
  const tools: Record<string, Tool> = {
    gather_context: tool({
      description: `Search memory, connected integrations, the web, AND connected gateways (user's devices like Claude Code, browser, etc.). This is how you access information.

      FOUR DATA SOURCES:
      1. Memory: past conversations, decisions, user preferences
      2. Integrations: user's emails, calendar, issues, messages (their personal data)
      3. Web: news, current events, documentation, prices, weather, general knowledge, AND reading URLs
      4. Gateways: user's connected devices/agents (e.g., Claude Code on their laptop, browser agent) - use for tasks on their machine

      WHEN TO USE:
      - Before saying "i don't know" - you might know it
      - When user asks about past conversations, decisions, preferences
      - When user asks about live data (emails, calendar, issues, etc.)
      - When user asks about news, current events, how-tos, or general questions
      - When user shares a URL and wants you to read/summarize it
      - When user asks to do something on their device/machine (coding tasks, file operations, browser actions)

      HOW TO FORM YOUR QUERY:
      Describe your INTENT clearly. Include any URLs the user shared.

      EXAMPLES:
      - "What meetings does user have this week" → integrations (calendar)
      - "What did we discuss about the deployment" → memory
      - "Latest tech news and AI updates" → web search
      - "What's the weather in SF" → web search
      - "Summarize this article: https://example.com/post" → web (fetches URL)
      - "User's unread emails from GitHub" → integrations (gmail)
      - "Check the status of user's local dev server" → gateway (connected device)

      For URLs: include the full URL in your query.
      For GENERAL NEWS/INFO: the orchestrator will use web search.
      For USER-SPECIFIC data: it uses integrations.
      For DEVICE/MACHINE tasks: it uses gateways.
      For SKILLS: when user's request matches an available skill, include the skill name and ID in your query so the orchestrator can load and execute it. Example: "Execute skill 'Plan My Day' (skill_id: abc123)"`,
      inputSchema: z.object({
        query: z
          .string()
          .describe(
            "Your intent - what you're looking for and why. Describe it like you're asking a colleague to find something.",
          ),
      }),

      execute: async function* ({ query }, { abortSignal }) {
        const toolRunId = crypto.randomUUID();
        const progressId = `gather-context:${toolRunId}`;
        const startedAt = Date.now();
        const progress = createToolProgressRelay();
        logger.info(`Core brain: Gathering context for: ${query}`);
        await progress.seed({
          id: progressId,
          level: 0,
          label: "Gather context",
          status: "running",
          detail: "Planning the context search",
          elapsedMs: 0,
        });
        yield { progress: progress.snapshot() };

        const { stream, progressId: orchestratorProgressId } =
          await runOrchestrator(
            userId,
            workspaceId,
            query,
            "read",
            timezone,
            source,
            abortSignal,
            persona,
            skills,
            progress.sink,
            progressId,
          );
        logger.info("Core brain: gather_context nested stream created", {
          toolRunId,
          elapsedMs: Date.now() - startedAt,
          queryChars: query.length,
        });

        try {
          const finalText = yield* progress.streamUntil(stream.text);
          await progress.sink({
            id: orchestratorProgressId ?? `orchestrator:${toolRunId}`,
            parentId: progressId,
            level: 1,
            label: "Orchestrator",
            status: finalText?.trim() ? "completed" : "failed",
            detail: finalText?.trim()
              ? "Final handoff summary ready"
              : "Final handoff summary unavailable",
            elapsedMs: Date.now() - startedAt,
          });
          if (!finalText?.trim()) {
            logger.warn("Core brain: gather_context produced empty final text", {
              toolRunId,
              elapsedMs: Date.now() - startedAt,
              progressEventCount: progress.snapshot().length,
            });
          }
          await progress.sink({
            id: progressId,
            level: 0,
            label: "Gather context",
            status: "completed",
            detail: "Context gathering complete",
            elapsedMs: Date.now() - startedAt,
          });
          yield compactNestedResult(
            finalText,
            "Context gathering completed.",
            progress.snapshot(),
          );
          logger.info("Core brain: gather_context nested stream closed", {
            toolRunId,
            elapsedMs: Date.now() - startedAt,
            finalTextChars: finalText?.length ?? 0,
          });
        } catch (error) {
          logger.error("Core brain: gather_context nested stream failed", {
            toolRunId,
            elapsedMs: Date.now() - startedAt,
            error,
          });
          throw error;
        }
      },
    }),
  };

  if (!readOnly) {
    tools["take_action"] = tool({
      description: `Execute actions on user's connected integrations AND gateways (connected devices).
      Use this to CREATE/SEND/UPDATE/DELETE: gmail filters/labels, calendar events, github issues, slack messages, notion pages.
      Also use this for tasks on user's connected devices/agents: coding tasks via Claude Code, browser actions, file operations on their machine.
      Examples: "post message to slack #team-updates saying deployment complete", "block friday 3pm on calendar for 1:1 with sarah", "create github issue in core repo titled fix auth timeout", "fix the auth bug in the core repo" (gateway task)
      When user confirms they want something done, use this tool to do it.
      For SKILLS: when executing a skill, include the skill name and ID. Example: "Execute skill 'Plan My Day' (skill_id: abc123)"`,
      inputSchema: z.object({
        action: z
          .string()
          .describe(
            "The action to perform. Be specific: include integration, what to create/send/update, and all details.",
          ),
      }),
      execute: async function* ({ action }, { abortSignal }) {
        const toolRunId = crypto.randomUUID();
        const progressId = `take-action:${toolRunId}`;
        const startedAt = Date.now();
        const progress = createToolProgressRelay();
        logger.info(`Core brain: Taking action: ${action}`);
        await progress.seed({
          id: progressId,
          level: 0,
          label: "Take action",
          status: "running",
          detail: "Planning the requested action",
          elapsedMs: 0,
        });
        yield { progress: progress.snapshot() };

        const { stream, progressId: orchestratorProgressId } =
          await runOrchestrator(
            userId,
            workspaceId,
            action,
            "write",
            timezone,
            source,
            abortSignal,
            persona,
            skills,
            progress.sink,
            progressId,
          );
        logger.info("Core brain: take_action nested stream created", {
          toolRunId,
          elapsedMs: Date.now() - startedAt,
          actionChars: action.length,
        });

        try {
          const finalText = yield* progress.streamUntil(stream.text);
          await progress.sink({
            id: orchestratorProgressId ?? `orchestrator:${toolRunId}`,
            parentId: progressId,
            level: 1,
            label: "Orchestrator",
            status: finalText?.trim() ? "completed" : "failed",
            detail: finalText?.trim()
              ? "Final handoff summary ready"
              : "Final handoff summary unavailable",
            elapsedMs: Date.now() - startedAt,
          });
          if (!finalText?.trim()) {
            logger.warn("Core brain: take_action produced empty final text", {
              toolRunId,
              elapsedMs: Date.now() - startedAt,
              progressEventCount: progress.snapshot().length,
            });
          }
          await progress.sink({
            id: progressId,
            level: 0,
            label: "Take action",
            status: "completed",
            detail: "Action complete",
            elapsedMs: Date.now() - startedAt,
          });
          yield compactNestedResult(
            finalText,
            "Action completed.",
            progress.snapshot(),
          );
          logger.info("Core brain: take_action nested stream closed", {
            toolRunId,
            elapsedMs: Date.now() - startedAt,
            finalTextChars: finalText?.length ?? 0,
          });
        } catch (error) {
          logger.error("Core brain: take_action nested stream failed", {
            toolRunId,
            elapsedMs: Date.now() - startedAt,
            error,
          });
          throw error;
        }
      },
    });
  }

  // Add acknowledge tool for channels with intermediate message support
  if (onMessage) {
    tools["acknowledge"] = tool({
      description:
        "Send a quick ack ONLY when you're about to call gather_context or take_action. Do NOT call this for simple greetings, thanks, or conversational messages - just respond directly for those.",
      inputSchema: z.object({
        message: z
          .string()
          .describe(
            'Brief ack referencing what you\'re about to look up. "checking your calendar." "pulling up your emails." "looking at your PRs." "on it." Keep it contextual.',
          ),
      }),
      execute: async ({ message }) => {
        logger.info(`Core brain: Acknowledging: ${message}`);
        await onMessage(message);
        return "acknowledged";
      },
    });
  }

  // Add reminder management tools
  // WhatsApp/Slack source → same channel, everything else (web/email) → email
  const channel =
    source === "whatsapp" ? "whatsapp" : source === "slack" ? "slack" : "email";
  const reminderTools = getReminderTools(workspaceId, channel, timezone);

  return { ...tools, ...reminderTools };
};
