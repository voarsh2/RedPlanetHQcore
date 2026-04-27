export type ToolProgressStatus = "running" | "completed" | "failed";

export interface ToolProgressEvent {
  id: string;
  parentId?: string;
  level?: number;
  label: string;
  status: ToolProgressStatus;
  detail?: string;
  elapsedMs?: number;
}

export type ToolProgressSink = (
  event: ToolProgressEvent,
) => void | Promise<void>;

const MAX_PROGRESS_EVENTS = 30;
const MAX_PROGRESS_DETAIL_CHARS = 240;

const trimDetail = (detail: string | undefined) => {
  if (!detail) return undefined;
  return detail.length > MAX_PROGRESS_DETAIL_CHARS
    ? `${detail.slice(0, MAX_PROGRESS_DETAIL_CHARS - 1)}...`
    : detail;
};

export const normalizeProgressEvent = (
  event: ToolProgressEvent,
): ToolProgressEvent => ({
  ...event,
  detail: trimDetail(event.detail),
});

export const mergeProgressEvent = (
  events: ToolProgressEvent[],
  event: ToolProgressEvent,
) => {
  const normalized = normalizeProgressEvent(event);
  const existingIndex = events.findIndex((item) => item.id === normalized.id);

  if (existingIndex >= 0) {
    const next = [...events];
    next[existingIndex] = { ...next[existingIndex], ...normalized };
    return next;
  }

  return [...events, normalized].slice(-MAX_PROGRESS_EVENTS);
};

export const compactNestedResult = (
  text: string | undefined,
  fallbackText: string,
  progress?: ToolProgressEvent[],
) => ({
  ...(progress?.length ? { progress } : {}),
  parts: [
    {
      type: "text",
      text: text?.trim() || fallbackText,
    },
  ],
});

export const compactProgressResult = (progress: ToolProgressEvent[]) => ({
  progress,
});

export async function withProgressHeartbeat<T>(
  work: PromiseLike<T>,
  {
    sink,
    event,
    startedAt,
    intervalMs = 15_000,
  }: {
    sink?: ToolProgressSink;
    event: Omit<ToolProgressEvent, "elapsedMs" | "status">;
    startedAt: number;
    intervalMs?: number;
  },
) {
  if (!sink) {
    return await work;
  }

  const timer = setInterval(() => {
    void sink({
      ...event,
      status: "running",
      elapsedMs: Date.now() - startedAt,
    });
  }, intervalMs);

  try {
    return await work;
  } finally {
    clearInterval(timer);
  }
}

export function createToolProgressRelay() {
  let events: ToolProgressEvent[] = [];
  const pendingSnapshots: ToolProgressEvent[][] = [];
  let wake: (() => void) | undefined;

  const snapshot = () => events;

  const sink: ToolProgressSink = (event) => {
    events = mergeProgressEvent(events, event);
    pendingSnapshots.push(events);
    wake?.();
    wake = undefined;
  };

  const seed: ToolProgressSink = (event) => {
    events = mergeProgressEvent(events, event);
  };

  const nextSnapshot = () => {
    if (pendingSnapshots.length > 0) {
      return Promise.resolve(pendingSnapshots.shift() ?? events);
    }

    return new Promise<ToolProgressEvent[]>((resolve) => {
      wake = () => resolve(pendingSnapshots.shift() ?? events);
    });
  };

  async function* streamUntil<T>(
    work: PromiseLike<T>,
  ): AsyncGenerator<{ progress: ToolProgressEvent[] }, T, void> {
    const workPromise = Promise.resolve(work);

    while (true) {
      const result = await Promise.race([
        workPromise.then(
          (value) => ({ type: "done" as const, value }),
          (error) => ({ type: "error" as const, error }),
        ),
        nextSnapshot().then((progress) => ({
          type: "progress" as const,
          progress,
        })),
      ]);

      if (result.type === "progress") {
        yield compactProgressResult(result.progress);
        continue;
      }

      if (result.type === "error") {
        throw result.error;
      }

      return result.value;
    }
  }

  return { seed, sink, snapshot, streamUntil };
}
