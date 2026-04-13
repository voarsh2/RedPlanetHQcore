export const CUSTOM_MCP_ENV_PREFIX = "MCP_";

export const CUSTOM_MCP_TRANSPORT_STRATEGIES = [
  "http-first",
  "sse-first",
  "http-only",
  "sse-only",
] as const;

export type CustomMcpTransportStrategy =
  (typeof CUSTOM_MCP_TRANSPORT_STRATEGIES)[number];

export interface CustomMcpHeaderConfig {
  name: string;
  value?: string;
  envKey?: string;
}

export interface CustomMcpOAuthConfig {
  accessToken?: string;
  refreshToken?: string;
  expiresIn?: number;
  clientId?: string;
  clientSecret?: string;
}

export interface CustomMcpIntegration {
  id: string;
  name: string;
  serverUrl: string;
  transportStrategy?: CustomMcpTransportStrategy;
  headers?: CustomMcpHeaderConfig[];
  oauth?: CustomMcpOAuthConfig;
}

export function parseCustomMcpHeadersInput(input: string): {
  headers: CustomMcpHeaderConfig[];
  error?: string;
} {
  const headers: CustomMcpHeaderConfig[] = [];

  for (const [index, rawLine] of input.split(/\r?\n/).entries()) {
    const line = rawLine.trim();
    if (!line) {
      continue;
    }

    const separatorIndex = line.indexOf("=");
    if (separatorIndex <= 0) {
      return {
        headers: [],
        error: `Invalid header on line ${index + 1}. Use Header-Name=value or Header-Name=env:${CUSTOM_MCP_ENV_PREFIX}KEY.`,
      };
    }

    const name = line.slice(0, separatorIndex).trim();
    const rawValue = line.slice(separatorIndex + 1).trim();

    if (!name || !rawValue) {
      return {
        headers: [],
        error: `Invalid header on line ${index + 1}. Header name and value are required.`,
      };
    }

    if (rawValue.startsWith("env:")) {
      const envKey = rawValue.slice(4).trim();
      if (!new RegExp(`^${CUSTOM_MCP_ENV_PREFIX}[A-Z0-9_]+$`).test(envKey)) {
        return {
          headers: [],
          error: `Invalid env key on line ${index + 1}. Only ${CUSTOM_MCP_ENV_PREFIX}* variables are allowed.`,
        };
      }

      headers.push({ name, envKey });
      continue;
    }

    headers.push({ name, value: rawValue });
  }

  return { headers };
}

export function stringifyCustomMcpHeaders(
  headers: CustomMcpHeaderConfig[] | undefined,
): string {
  if (!headers || headers.length === 0) {
    return "";
  }

  return headers
    .map((header) =>
      header.envKey
        ? `${header.name}=env:${header.envKey}`
        : `${header.name}=${header.value ?? ""}`,
    )
    .join("\n");
}

export function resolveCustomMcpHeaders(
  headers: CustomMcpHeaderConfig[] | undefined,
  envSource?: Record<string, string | undefined>,
): Record<string, string> {
  if (!headers || headers.length === 0) {
    return {};
  }

  const source =
    envSource ??
    (typeof process !== "undefined" ? process.env : ({} as Record<string, string | undefined>));
  const resolvedHeaders: Record<string, string> = {};

  for (const header of headers) {
    if (header.envKey) {
      const envValue = source[header.envKey];
      if (envValue) {
        resolvedHeaders[header.name] = envValue;
      }
      continue;
    }

    if (header.value) {
      resolvedHeaders[header.name] = header.value;
    }
  }

  return resolvedHeaders;
}
