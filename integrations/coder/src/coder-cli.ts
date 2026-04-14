import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);

const CODER_TIMEOUT_MS = 120_000;
const CODER_MAX_BUFFER = 10 * 1024 * 1024;

export interface CoderIntegrationConfig {
  coderUrl: string;
  sessionToken: string;
}

export function getCoderConfig(rawConfig: Record<string, unknown> = {}): CoderIntegrationConfig {
  const coderUrl = String(rawConfig.coderUrl || process.env.CODER_URL || '').trim();
  const sessionToken = String(
    rawConfig.sessionToken || process.env.CODER_SESSION_TOKEN || ''
  ).trim();

  if (!coderUrl) {
    throw new Error('Coder integration is missing coderUrl in integration config.');
  }

  if (!sessionToken) {
    throw new Error('Coder integration is missing sessionToken in integration config.');
  }

  return {
    coderUrl,
    sessionToken,
  };
}

function buildCoderEnv(config: CoderIntegrationConfig) {
  return {
    ...process.env,
    CODER_URL: config.coderUrl,
    CODER_SESSION_TOKEN: config.sessionToken,
    CODER_NO_VERSION_WARNING: 'true',
  };
}

export async function runCoderCommand(
  args: string[],
  rawConfig: Record<string, unknown> = {}
) {
  const config = getCoderConfig(rawConfig);

  try {
    return await execFileAsync('coder', args, {
      env: buildCoderEnv(config),
      encoding: 'utf-8',
      timeout: CODER_TIMEOUT_MS,
      maxBuffer: CODER_MAX_BUFFER,
    });
  } catch (error: any) {
    const stderr = error?.stderr?.toString?.().trim?.() || '';
    const stdout = error?.stdout?.toString?.().trim?.() || '';
    const details = [stderr, stdout].filter(Boolean).join('\n');
    throw new Error(
      `Coder command failed: coder ${args.join(' ')}${details ? `\n${details}` : ''}`
    );
  }
}

export async function runCoderJson<T = unknown>(
  args: string[],
  rawConfig: Record<string, unknown> = {}
): Promise<T> {
  const { stdout } = await runCoderCommand(args, rawConfig);

  try {
    return JSON.parse(stdout) as T;
  } catch (error) {
    throw new Error(`Failed to parse JSON from coder ${args.join(' ')}: ${stdout}`);
  }
}
