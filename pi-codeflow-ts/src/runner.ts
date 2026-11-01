/** Pi-backed implementation of the CodeFlow TurnRunner boundary. */

import { createAgentSession, SessionManager } from "@earendil-works/pi-coding-agent";
import { ContextBudgetManager } from "./context.js";
import { ToolPolicyRegistry, type ToolEffect } from "./policy.js";
import type { Drain, Emit, TurnOutcome, TurnRequest, TurnRunner } from "./protocol.js";

type PiSession = Awaited<ReturnType<typeof createAgentSession>>["session"];

interface ManagedPiSession {
  readonly workspace: string;
  readonly session: PiSession;
  readonly prepareRequest: PiSession["agent"]["prepareRequest"];
  readonly beforeToolCall: PiSession["agent"]["beforeToolCall"];
}

export class PiTurnRunner implements TurnRunner {
  readonly #contextBudget: ContextBudgetManager;
  readonly #toolPolicies: ToolPolicyRegistry;
  readonly #allowedEffects: ReadonlySet<ToolEffect>;
  readonly #sessions = new Map<string, ManagedPiSession>();
  readonly #sessionCreations = new Map<string, Promise<ManagedPiSession>>();

  constructor(options: {
    contextTokenEstimate?: number;
    allowedEffects?: readonly ToolEffect[];
    toolEffects?: Readonly<Record<string, ToolEffect>>;
  } = {}) {
    this.#contextBudget = new ContextBudgetManager(options.contextTokenEstimate ?? 12_000, 8);
    this.#toolPolicies = new ToolPolicyRegistry();
    for (const [toolName, effect] of Object.entries(options.toolEffects ?? {})) {
      this.#toolPolicies.register(toolName, effect);
    }
    this.#allowedEffects = new Set(options.allowedEffects ?? ["read"]);
  }

  async run(request: TurnRequest, emit: Emit, _drain: Drain, signal?: AbortSignal): Promise<TurnOutcome> {
    const managed = await this.#getSession(request.conversationId, request.workspace);
    const { session } = managed;
    if (session.isStreaming) throw new Error(`Pi conversation is already running: ${request.conversationId}`);
    let toolCalls = 0;
    let toolFailures = 0;
    let explicitReply = false;
    let cancelled = false;
    let runtimeFailed = false;
    let inputTokens = 0;
    let outputTokens = 0;
    let trimmedMessages = 0;
    let contextPruneReason: TurnOutcome["contextPruneReason"] = "within_budget";
    const recordBlockedTool = async (toolCallId: string, name: string, args: unknown) => {
      toolCalls++;
      toolFailures++;
      await emit({ type: "tool_start", callId: toolCallId, name, arguments: args });
      await emit({ type: "tool_end", callId: toolCallId, name, failed: true, durationMs: 0 });
    };
    session.agent.prepareRequest = async (request, requestSignal) => {
      const prior = await managed.prepareRequest?.(request, requestSignal);
      const context = prior?.context ?? request.context;
      const result = this.#contextBudget.apply(context.messages);
      trimmedMessages += result.trimmedMessages;
      contextPruneReason = result.reason;
      return { ...prior, context: { ...context, messages: [...result.messages] } };
    };
    session.agent.beforeToolCall = async (callContext, callSignal) => {
      const { toolCall } = callContext;
      const decision = this.#toolPolicies.decide(toolCall.name, this.#allowedEffects);
      if (!decision.allowed) {
        await recordBlockedTool(toolCall.id, toolCall.name, callContext.args);
        return { block: true, ...(decision.reason ? { reason: decision.reason } : {}) };
      }
      const prior = await managed.beforeToolCall?.(callContext, callSignal);
      if (prior?.block) await recordBlockedTool(toolCall.id, toolCall.name, callContext.args);
      return prior;
    };
    const onAbort = () => {
      cancelled = true;
      void session.abort();
    };
    signal?.addEventListener("abort", onAbort, { once: true });

    const unsubscribe = session.subscribe(async (event) => {
      if (event.type === "message_update" && event.assistantMessageEvent.type === "text_delta") {
        explicitReply ||= Boolean(event.assistantMessageEvent.delta);
        await emit({ type: "text_delta", delta: event.assistantMessageEvent.delta });
      } else if (event.type === "message_end" && event.message.role === "assistant") {
        if (event.message.stopReason === "error" || event.message.stopReason === "aborted") runtimeFailed = true;
        inputTokens += event.message.usage.input;
        outputTokens += event.message.usage.output;
      } else if (event.type === "tool_execution_start") {
        toolCalls++;
        await emit({
          type: "tool_start",
          callId: event.toolCallId,
          name: event.toolName,
          arguments: event.args,
        });
      } else if (event.type === "tool_execution_end") {
        const failed = Boolean(event.isError);
        if (failed) toolFailures++;
        await emit({
          type: "tool_end",
          callId: event.toolCallId,
          name: event.toolName,
          failed,
          durationMs: event.durationMs ?? 0,
        });
      }
    });

    try {
      if (signal?.aborted) return { status: "cancelled", explicitReply, toolCalls, toolFailures };
      await session.prompt(request.text);
      if (trimmedMessages > 0) {
        await emit({ type: "notice", message: `Context budget trimmed ${trimmedMessages} older messages.` });
      }
      return {
        status: cancelled ? "cancelled" : runtimeFailed ? "error" : "completed",
        explicitReply,
        toolCalls,
        toolFailures,
        usage: { inputTokens, outputTokens },
        contextPrunedMessages: trimmedMessages,
        contextPruneReason,
        ...(runtimeFailed ? { errorCategory: "pi_provider_error" } : {}),
      };
    } catch (error) {
      if (cancelled || signal?.aborted) {
        return { status: "cancelled", explicitReply, toolCalls, toolFailures };
      }
      await emit({ type: "notice", message: error instanceof Error ? error.message : String(error) });
      return { status: "error", explicitReply, toolCalls, toolFailures, errorCategory: "pi_run_failed" };
    } finally {
      unsubscribe();
      signal?.removeEventListener("abort", onAbort);
      if (managed.prepareRequest) session.agent.prepareRequest = managed.prepareRequest;
      else delete session.agent.prepareRequest;
      if (managed.beforeToolCall) session.agent.beforeToolCall = managed.beforeToolCall;
      else delete session.agent.beforeToolCall;
    }
  }

  async closeConversation(conversationId: string): Promise<void> {
    const managed = this.#sessions.get(conversationId);
    if (!managed) return;
    await managed.session.abort();
    managed.session.dispose();
    this.#sessions.delete(conversationId);
  }

  async close(): Promise<void> {
    await Promise.all([...this.#sessions.keys()].map((conversationId) => this.closeConversation(conversationId)));
  }

  async #getSession(conversationId: string, workspace: string): Promise<ManagedPiSession> {
    const existing = this.#sessions.get(conversationId);
    if (existing) {
      if (existing.workspace !== workspace) throw new Error("Pi conversation cannot change workspace");
      return existing;
    }
    const pending = this.#sessionCreations.get(conversationId);
    if (pending) {
      const managed = await pending;
      if (managed.workspace !== workspace) throw new Error("Pi conversation cannot change workspace");
      return managed;
    }

    const creation = this.#createSession(workspace);
    this.#sessionCreations.set(conversationId, creation);
    try {
      const managed = await creation;
      this.#sessions.set(conversationId, managed);
      return managed;
    } finally {
      if (this.#sessionCreations.get(conversationId) === creation) this.#sessionCreations.delete(conversationId);
    }
  }

  async #createSession(workspace: string): Promise<ManagedPiSession> {
    const { session } = await createAgentSession({ cwd: workspace, sessionManager: SessionManager.inMemory() });
    return {
      workspace,
      session,
      prepareRequest: session.agent.prepareRequest,
      beforeToolCall: session.agent.beforeToolCall,
    };
  }
}
