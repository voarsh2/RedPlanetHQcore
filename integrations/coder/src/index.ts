import {
  IntegrationCLI,
  IntegrationEventPayload,
  IntegrationEventType,
  Spec,
} from '@redplanethq/sdk';

import { integrationCreate } from './account-create';
import { callTool, getTools } from './mcp';

export async function run(eventPayload: IntegrationEventPayload) {
  switch (eventPayload.event) {
    case IntegrationEventType.SETUP:
      return await integrationCreate(
        eventPayload.eventBody as Record<string, string>,
        (eventPayload.integrationDefinition?.config as Record<string, unknown>) || {}
      );

    case IntegrationEventType.GET_TOOLS:
      return await getTools();

    case IntegrationEventType.CALL_TOOL: {
      const integrationDefinition = eventPayload.integrationDefinition;

      if (!integrationDefinition) {
        return null;
      }

      const { name, arguments: args } = eventPayload.eventBody;

      return await callTool(
        name,
        args,
        (integrationDefinition.config as Record<string, unknown>) || {}
      );
    }

    default:
      return { message: `The event payload type is ${eventPayload.event}` };
  }
}

class CoderCLI extends IntegrationCLI {
  constructor() {
    super('coder', '1.0.0');
  }

  protected async handleEvent(eventPayload: IntegrationEventPayload): Promise<any> {
    return await run(eventPayload);
  }

  protected async getSpec(): Promise<Spec> {
    return {
      name: 'Coder extension',
      key: 'coder',
      description:
        'Connect your workspace to Coder. Inspect workspaces, start or stop them, and run commands inside real remote development environments.',
      icon: 'code',
      mcp: {
        type: 'cli',
      },
      auth: {
        api_key: {
          label: 'Server-managed credentials',
          description:
            'Uses CODER_URL and CODER_SESSION_TOKEN from the CORE server environment.',
          allowEmpty: true,
          connectLabel: 'Connect using server credentials',
        },
      },
    };
  }
}

function main() {
  const coderCLI = new CoderCLI();
  coderCLI.parse();
}

main();
