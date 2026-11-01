/** TypeScript port of CodeFlow's Spine request, runner event and terminal outcome contracts. */

export type TurnOrigin = "user" | "cron" | "subagent" | "system";

export interface TurnRequest {
  readonly turnId: string;
  readonly conversationId: string;
  readonly text: string;
  readonly origin: TurnOrigin;
  readonly workspace: string;
}

/** Synchronous hook for retrieving pending same-lane requests between loop steps. */
export type Drain = () => TurnRequest[];

export type RunnerEvent =
  | { readonly type: "text_delta"; readonly delta: string }
  | { readonly type: "reasoning"; readonly content: string }
  | { readonly type: "tool_start"; readonly callId: string; readonly name: string; readonly arguments: unknown }
  | { readonly type: "tool_end"; readonly callId: string; readonly name: string; readonly failed: boolean; readonly durationMs: number }
  | { readonly type: "notice"; readonly message: string };

export interface TurnOutcome {
  readonly status: "completed" | "interrupted" | "error" | "cancelled";
  readonly explicitReply: boolean;
  readonly toolCalls: number;
  readonly toolFailures: number;
  readonly contextPrunedMessages?: number;
  readonly contextPruneReason?: "within_budget" | "oldest_unprotected_trimmed" | "protected_context_exceeds_budget";
  readonly usage?: { readonly inputTokens: number; readonly outputTokens: number };
  readonly errorCategory?: string;
}

export type Emit = (event: RunnerEvent) => Promise<void>;

export interface TurnRunner {
  run(request: TurnRequest, emit: Emit, drain: Drain, signal?: AbortSignal): Promise<TurnOutcome>;
}
