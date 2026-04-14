import { getCoderConfig, runCoderJson } from './coder-cli';

interface CoderWhoAmI {
  id?: string;
  username?: string;
  email?: string;
  url?: string;
  organizations?: Array<{ name?: string }>;
}

export async function integrationCreate(
  data: Record<string, string>,
  integrationConfig: Record<string, unknown> = {}
) {
  const coderConfig = getCoderConfig(integrationConfig);
  const whoami = await runCoderJson<CoderWhoAmI>(
    ['whoami', '--output', 'json'],
    coderConfig
  );

  const username = whoami.username || 'coder-user';
  const connectedUserId = String(data.userId || '').trim();
  const serviceIdentity = whoami.id || username;
  const accountId = connectedUserId
    ? `${coderConfig.coderUrl}:${serviceIdentity}:${connectedUserId}`
    : `${coderConfig.coderUrl}:${serviceIdentity}`;

  return [
    {
      type: 'account',
      data: {
        settings: {
          coder: {
            url: whoami.url || coderConfig.coderUrl,
            username,
            email: whoami.email || null,
            organizations: whoami.organizations || [],
          },
        },
        accountId,
        config: {
          coderUrl: coderConfig.coderUrl,
        },
      },
    },
  ];
}
