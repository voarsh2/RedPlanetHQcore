import { z } from 'zod';
import { zodToJsonSchema } from 'zod-to-json-schema';

import { runCoderCommand, runCoderJson } from '../coder-cli';

const EmptySchema = z.object({});

const ListWorkspacesSchema = z.object({
  all: z.boolean().optional().default(false).describe('Include all visible workspaces'),
  search: z
    .string()
    .optional()
    .describe('Optional coder search query. Defaults to owner:me when omitted.'),
});

const ListTemplatesSchema = z.object({});

const ShowWorkspaceSchema = z.object({
  workspace: z.string().describe('Workspace name'),
  details: z.boolean().optional().default(false).describe('Show extra details if available'),
});

const StartWorkspaceSchema = z.object({
  workspace: z.string().describe('Workspace name'),
  waitForReady: z.boolean().optional().default(true).describe('Wait for startup to finish'),
});

const StopWorkspaceSchema = z.object({
  workspace: z.string().describe('Workspace name'),
});

const ExecWorkspaceCommandSchema = z.object({
  workspace: z.string().describe('Workspace name'),
  command: z.string().describe('Shell command to run inside the workspace'),
  autostart: z
    .boolean()
    .optional()
    .default(true)
    .describe('Allow Coder to start the workspace automatically before exec'),
});

export async function getTools() {
  return [
    {
      name: 'coder_whoami',
      description:
        'Show the authenticated Coder user and organization access for the configured automation account.',
      inputSchema: zodToJsonSchema(EmptySchema),
      annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true },
    },
    {
      name: 'coder_list_workspaces',
      description:
        'List Coder workspaces visible to the configured automation account, including status, health, template, owner, and organization.',
      inputSchema: zodToJsonSchema(ListWorkspacesSchema),
      annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true },
    },
    {
      name: 'coder_list_templates',
      description:
        'List Coder templates available to the configured automation account for workspace creation and updates.',
      inputSchema: zodToJsonSchema(ListTemplatesSchema),
      annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true },
    },
    {
      name: 'coder_show_workspace',
      description:
        'Show detailed information for one Coder workspace, including resources and agents when available.',
      inputSchema: zodToJsonSchema(ShowWorkspaceSchema),
      annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true },
    },
    {
      name: 'coder_start_workspace',
      description: 'Start a Coder workspace.',
      inputSchema: zodToJsonSchema(StartWorkspaceSchema),
      annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: true },
    },
    {
      name: 'coder_stop_workspace',
      description: 'Stop a Coder workspace.',
      inputSchema: zodToJsonSchema(StopWorkspaceSchema),
      annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: true },
    },
    {
      name: 'coder_exec_workspace_command',
      description: 'Run a shell command inside a Coder workspace over coder ssh.',
      inputSchema: zodToJsonSchema(ExecWorkspaceCommandSchema),
      annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: false },
    },
  ];
}

function workspaceSummary(workspace: any) {
  return [
    `Name: ${workspace.name || 'unknown'}`,
    workspace.owner_name ? `Owner: ${workspace.owner_name}` : null,
    workspace.organization_name ? `Organization: ${workspace.organization_name}` : null,
    workspace.template_name ? `Template: ${workspace.template_name}` : null,
    workspace.latest_build?.status ? `Status: ${workspace.latest_build.status}` : null,
    workspace.latest_build?.transition ? `Transition: ${workspace.latest_build.transition}` : null,
    workspace.health?.healthy === true
      ? 'Healthy: yes'
      : workspace.health?.healthy === false
        ? 'Healthy: no'
        : null,
    workspace.outdated === true ? 'Outdated: yes' : null,
  ]
    .filter(Boolean)
    .join('\n');
}

function templateSummary(template: any) {
  return [
    `Name: ${template.name || 'unknown'}`,
    template.organization_name ? `Organization: ${template.organization_name}` : null,
    template.display_name ? `Display Name: ${template.display_name}` : null,
    template.default_ttl_ms
      ? `Default TTL (ms): ${template.default_ttl_ms}`
      : null,
    template.active_version_id ? `Active Version ID: ${template.active_version_id}` : null,
    typeof template.active_user_count === 'number'
      ? `Used By: ${template.active_user_count}`
      : null,
  ]
    .filter(Boolean)
    .join('\n');
}

export async function callTool(
  name: string,
  args: Record<string, unknown>,
  integrationConfig: Record<string, unknown> = {}
) {
  switch (name) {
    case 'coder_whoami': {
      EmptySchema.parse(args);
      const identities = await runCoderJson<any[]>(['whoami', '--output', 'json'], integrationConfig);

      return {
        content: [
          {
            type: 'text',
            text:
              identities.length === 0
                ? 'No authenticated Coder identities were returned.'
                : identities
                    .map((identity) =>
                      [
                        `URL: ${identity.url || 'unknown'}`,
                        identity.username ? `Username: ${identity.username}` : null,
                        identity.user_id ? `User ID: ${identity.user_id}` : null,
                        Array.isArray(identity.organization_ids) && identity.organization_ids.length > 0
                          ? `Organization IDs: ${identity.organization_ids.join(', ')}`
                          : null,
                      ]
                        .filter(Boolean)
                        .join('\n')
                    )
                    .join('\n\n'),
          },
        ],
      };
    }

    case 'coder_list_workspaces': {
      const { all, search } = ListWorkspacesSchema.parse(args);
      const commandArgs = ['list', '--output', 'json'];

      if (all) {
        commandArgs.push('--all');
      }

      commandArgs.push('--search', search?.trim() || 'owner:me');

      const workspaces = await runCoderJson<any[]>(commandArgs, integrationConfig);

      if (workspaces.length === 0) {
        return {
          content: [{ type: 'text', text: 'No Coder workspaces found.' }],
        };
      }

      return {
        content: [
          {
            type: 'text',
            text:
              `Found ${workspaces.length} Coder workspace${workspaces.length === 1 ? '' : 's'}:\n\n` +
              workspaces.map(workspaceSummary).join('\n\n'),
          },
        ],
      };
    }

    case 'coder_list_templates': {
      ListTemplatesSchema.parse(args);
      const templates = await runCoderJson<any[]>(
        ['templates', 'list', '--output', 'json'],
        integrationConfig
      );

      if (templates.length === 0) {
        return {
          content: [{ type: 'text', text: 'No Coder templates found.' }],
        };
      }

      return {
        content: [
          {
            type: 'text',
            text:
              `Found ${templates.length} Coder template${templates.length === 1 ? '' : 's'}:\n\n` +
              templates.map(templateSummary).join('\n\n'),
          },
        ],
      };
    }

    case 'coder_show_workspace': {
      const { workspace, details } = ShowWorkspaceSchema.parse(args);
      const commandArgs = ['show'];

      if (details) {
        commandArgs.push('--details');
      }

      commandArgs.push(workspace);

      const { stdout } = await runCoderCommand(commandArgs, integrationConfig);

      return {
        content: [
          {
            type: 'text',
            text: stdout.trim() || `No details returned for workspace ${workspace}.`,
          },
        ],
      };
    }

    case 'coder_start_workspace': {
      const { workspace, waitForReady } = StartWorkspaceSchema.parse(args);
      const commandArgs = ['start', workspace, '--yes'];

      if (!waitForReady) {
        commandArgs.push('--no-wait');
      }

      const { stdout } = await runCoderCommand(commandArgs, integrationConfig);

      return {
        content: [
          {
            type: 'text',
            text: stdout.trim() || `Start requested for workspace ${workspace}.`,
          },
        ],
      };
    }

    case 'coder_stop_workspace': {
      const { workspace } = StopWorkspaceSchema.parse(args);
      const { stdout } = await runCoderCommand(
        ['stop', workspace, '--yes'],
        integrationConfig
      );

      return {
        content: [
          {
            type: 'text',
            text: stdout.trim() || `Stop requested for workspace ${workspace}.`,
          },
        ],
      };
    }

    case 'coder_exec_workspace_command': {
      const { workspace, command, autostart } = ExecWorkspaceCommandSchema.parse(args);
      const commandArgs = ['ssh'];

      if (!autostart) {
        commandArgs.push('--disable-autostart');
      }

      commandArgs.push(workspace, '--', 'bash', '-lc', command);

      const { stdout, stderr } = await runCoderCommand(commandArgs, integrationConfig);

      const sections = [
        `Workspace: ${workspace}`,
        `Command: ${command}`,
        stdout.trim() ? `STDOUT:\n${stdout.trim()}` : null,
        stderr.trim() ? `STDERR:\n${stderr.trim()}` : null,
      ].filter(Boolean);

      return {
        content: [
          {
            type: 'text',
            text: sections.join('\n\n'),
          },
        ],
      };
    }

    default:
      throw new Error(`Unknown Coder tool: ${name}`);
  }
}
