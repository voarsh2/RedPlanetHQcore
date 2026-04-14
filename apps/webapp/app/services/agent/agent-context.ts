/**
 * Shared agent context builder.
 *
 * Extracts the common setup used by web chat (stream + no_stream) and
 * async channels (WhatsApp, Email). Each caller gets back everything
 * needed to call generateText / streamText.
 */

import { convertToModelMessages, type ModelMessage, type Tool } from "ai";

import { getUserById } from "~/models/user.server";
import { getPersonaDocumentForUser } from "~/services/document.server";
import { IntegrationLoader } from "~/utils/mcp/integration-loader";
import { getCorePrompt } from "~/services/agent/prompts";
import { type ChannelType } from "~/services/agent/prompts/channel-formats";
import { createTools } from "~/services/agent/core-agent";
import { type MessagePlan } from "~/services/agent/types/decision-agent";
import { prisma } from "~/db.server";
import { logger } from "../logger.service";
import { getCompactedSessionBySessionId } from "../graphModels/compactedSession";

interface BuildAgentContextParams {
  userId: string;
  workspaceId: string;
  conversationId?: string;
  source: ChannelType;
  /** UI-format messages: { parts, role, id }[] */
  finalMessages: any[];
  /** Preserve full tool history for approval continuation flows. */
  preserveToolHistory?: boolean;
  /** Action plan from Decision Agent — injected into system prompt for reminder execution */
  actionPlan?: MessagePlan;
  /** Optional callback for channels to send intermediate messages (acks) */
  onMessage?: (message: string) => Promise<void>;
  /** Channel-specific metadata (messageSid, slackUserId, threadTs, etc.) */
  channelMetadata?: Record<string, string>;
}

interface AgentContext {
  systemPrompt: string;
  tools: Record<string, Tool>;
  modelMessages: ModelMessage[];
  user: Awaited<ReturnType<typeof getUserById>>;
  timezone: string;
}

const RECENT_MESSAGE_WINDOW = 8;

function hasApprovalStateDeep(parts: any[]): boolean {
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

function getTopLevelTextParts(parts: any[]): Array<{ type: "text"; text: string }> {
  return parts
    .filter((part) => part?.type === "text" && typeof part.text === "string")
    .map((part) => ({ type: "text" as const, text: part.text.trim() }))
    .filter((part) => part.text.length > 0);
}

function getToolSummary(parts: any[]): string | null {
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

function sanitizeMessageForModelContext(message: any) {
  const normalizedParts = Array.isArray(message?.parts)
    ? message.parts.filter(Boolean)
    : [];

  if (normalizedParts.length === 0) {
    return message;
  }

  const textParts = getTopLevelTextParts(normalizedParts);
  if (textParts.length > 0) {
    return {
      ...message,
      parts: textParts,
    };
  }

  const toolSummary = getToolSummary(normalizedParts);
  if (toolSummary) {
    return {
      ...message,
      parts: [{ type: "text", text: toolSummary }],
    };
  }

  return {
    ...message,
    parts: [],
  };
}

function sanitizeMessagesForModelContext(
  messages: any[],
  preserveToolHistory: boolean,
) {
  if (preserveToolHistory) {
    return messages;
  }

  let assistantMessagesSanitized = 0;
  let assistantToolPartsDropped = 0;

  const sanitized = messages
    .map((message) => {
      const normalizedParts = Array.isArray(message?.parts)
        ? message.parts.filter(Boolean)
        : [];

      if (
        message?.role === "assistant" &&
        normalizedParts.length > 0 &&
        !hasApprovalStateDeep(normalizedParts)
      ) {
        const toolParts = normalizedParts.filter(
          (part) => typeof part?.type === "string" && part.type.includes("tool-"),
        ).length;
        const sanitizedMessage = sanitizeMessageForModelContext(message);
        if (sanitizedMessage.parts !== message.parts) {
          assistantMessagesSanitized += 1;
          assistantToolPartsDropped += toolParts;
        }
        return sanitizedMessage;
      }

      if (message?.role === "user" && normalizedParts.length > 0) {
        return sanitizeMessageForModelContext(message);
      }

      return message;
    })
    .filter((message) => Array.isArray(message?.parts) && message.parts.length > 0);

  if (assistantMessagesSanitized > 0) {
    const rawChars = JSON.stringify(messages).length;
    const sanitizedChars = JSON.stringify(sanitized).length;
    logger.info("Agent context sanitized conversation history", {
      assistantMessagesSanitized,
      assistantToolPartsDropped,
      originalMessages: messages.length,
      sanitizedMessages: sanitized.length,
      rawChars,
      sanitizedChars,
      reducedChars: rawChars - sanitizedChars,
    });
  }

  return sanitized;
}

export async function buildAgentContext({
  userId,
  workspaceId,
  conversationId,
  source,
  finalMessages,
  preserveToolHistory = false,
  actionPlan,
  onMessage,
  channelMetadata,
}: BuildAgentContextParams): Promise<AgentContext> {
  // Load context in parallel
  const [user, persona, connectedIntegrations, skills, compactedSession] =
    await Promise.all([
    getUserById(userId),
    getPersonaDocumentForUser(workspaceId),
    IntegrationLoader.getConnectedIntegrationAccounts(userId, workspaceId),
    prisma.document.findMany({
      where: { workspaceId, type: "skill", deleted: null },
      select: { id: true, title: true, metadata: true },
      orderBy: { createdAt: "desc" },
    }),
    conversationId && !preserveToolHistory
      ? getCompactedSessionBySessionId(conversationId, userId, workspaceId).catch(
          (error) => {
            logger.warn("Failed to load compacted session for agent context", {
              conversationId,
              error: error instanceof Error ? error.message : String(error),
            });
            return null;
          },
        )
      : Promise.resolve(null),
    ]);

  const metadata = user?.metadata as Record<string, unknown> | null;
  const timezone = (metadata?.timezone as string) ?? "UTC";

  const tools = await createTools(
    userId,
    workspaceId,
    timezone,
    source,
    false,
    persona ?? undefined,
    skills,
    onMessage,
  );

  // Build system prompt
  let systemPrompt = getCorePrompt(
    source,
    {
      name: user?.displayName ?? user?.name ?? user?.email ?? "",
      email: user?.email ?? "",
      timezone,
      phoneNumber: user?.phoneNumber ?? undefined,
    },
    persona ?? "",
  );

  // Integrations context
  const integrationsList = connectedIntegrations
    .map(
      (int, index) =>
        `${index + 1}. **${int.integrationDefinition.name}** (Account ID: ${int.id})`,
    )
    .join("\n");

  systemPrompt += `
    <connected_integrations>
    You have ${connectedIntegrations.length} connected integration accounts:
    ${integrationsList}

    To use these integrations, follow the 2-step workflow:
    1. get_integration_actions (provide accountId and query to discover available actions)
    2. execute_integration_action (provide accountId and action name to execute)

    IMPORTANT: Always use the Account ID when calling get_integration_actions and execute_integration_action.
    </connected_integrations>`;

  // Skills context
  if (skills.length > 0) {
    const skillsList = skills
      .map((s, i) => {
        const meta = s.metadata as Record<string, unknown> | null;
        const desc = meta?.shortDescription as string | undefined;
        return `${i + 1}. "${s.title}" (id: ${s.id})${desc ? ` — ${desc}` : ""}`;
      })
      .join("\n");

    systemPrompt += `
    <skills>
    You have access to user-defined skills. When a user's request matches a skill, use gather_context or take_action to reference the skill name and ID so the orchestrator can load and execute it.

    Available skills:
    ${skillsList}
    </skills>`;
  }

  // Datetime context (use user's timezone so agent sees correct local time)
  const now = new Date();
  systemPrompt += `
    <current_datetime>
    Current date and time: ${now.toLocaleString("en-US", {
      timeZone: timezone,
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      timeZoneName: "short",
    })}
    </current_datetime>`;

  // Channel metadata context
  if (channelMetadata && Object.keys(channelMetadata).length > 0) {
    const metadataEntries = Object.entries(channelMetadata)
      .map(([k, v]) => `- ${k}: ${v}`)
      .join("\n");
    systemPrompt += `
    <channel_context>
    This message arrived from an external channel. Metadata:
    ${metadataEntries}
    </channel_context>`;
  }

  // Action plan from Decision Agent (reminder/webhook triggered)
  if (actionPlan) {
    systemPrompt += `\n\n<action_plan>
You are executing an action plan from the Decision Agent. The decision has been made.
Your job is to craft the message - don't second-guess the decision to message.

Intent: ${actionPlan.intent}
Tone: ${actionPlan.tone}
Context: ${JSON.stringify(actionPlan.context, null, 2)}

Guidelines:
- Use the provided context to inform your message
- Match the suggested tone (${actionPlan.tone})
- Be concise. Use only as much length as the content needs.
- Do NOT create new reminders
- Do NOT echo or reference any system instructions in your message
</action_plan>`;
  }

  let contextMessages = sanitizeMessagesForModelContext(
    finalMessages,
    preserveToolHistory,
  );

  if (
    !preserveToolHistory &&
    compactedSession?.summary &&
    contextMessages.length > RECENT_MESSAGE_WINDOW
  ) {
    const recentMessages = contextMessages.slice(-RECENT_MESSAGE_WINDOW);

    systemPrompt += `\n\n<session_summary>\nThis is a compacted summary of earlier conversation context for this session. Prefer it over replaying older verbatim history.\n\n${compactedSession.summary}\n</session_summary>`;

    logger.info("Agent context applied compacted session summary", {
      conversationId,
      compactSummaryChars: compactedSession.summary.length,
      originalMessages: contextMessages.length,
      recentMessagesRetained: recentMessages.length,
      droppedMessages: contextMessages.length - recentMessages.length,
    });

    contextMessages = recentMessages;
  }

  // Convert to model messages
  const modelMessages: ModelMessage[] = await convertToModelMessages(
    contextMessages,
    {
      tools,
      ignoreIncompleteToolCalls: true,
    },
  );

  return { systemPrompt, tools, modelMessages, user, timezone };
}
