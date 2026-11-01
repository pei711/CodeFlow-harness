/** Budget Pi transcript messages without breaking assistant-tool-result groups. */

import type { AgentMessage } from "@earendil-works/pi-agent-core";

export interface ContextResult {
  readonly messages: readonly AgentMessage[];
  readonly estimatedTokensBefore: number;
  readonly estimatedTokensAfter: number;
  readonly trimmedMessages: number;
  readonly reason: "within_budget" | "oldest_unprotected_trimmed" | "protected_context_exceeds_budget";
}

export class ContextBudgetManager {
  readonly maxTokenEstimate: number;
  readonly keepRecent: number;

  constructor(maxTokenEstimate: number, keepRecent: number = 8) {
    if (!Number.isInteger(maxTokenEstimate) || maxTokenEstimate < 1) throw new Error("maxTokenEstimate must be a positive integer");
    if (!Number.isInteger(keepRecent) || keepRecent < 1) throw new Error("keepRecent must be a positive integer");
    this.maxTokenEstimate = maxTokenEstimate;
    this.keepRecent = keepRecent;
  }

  apply(messages: readonly AgentMessage[]): ContextResult {
    const cost = (items: readonly AgentMessage[]) => items.reduce((n, item) => n + estimateMessageTokens(item), 0);
    const estimatedTokensBefore = cost(messages);
    if (estimatedTokensBefore <= this.maxTokenEstimate) {
      return { messages, estimatedTokensBefore, estimatedTokensAfter: estimatedTokensBefore, trimmedMessages: 0, reason: "within_budget" };
    }

    const recentStart = Math.max(0, messages.length - this.keepRecent);
    const removableGroups = findRemovableGroups(messages, recentStart);
    const removed = new Set<number>();
    let remainingTokens = estimatedTokensBefore;
    for (const group of removableGroups) {
      if (remainingTokens <= this.maxTokenEstimate) break;
      for (let index = group.start; index <= group.end; index++) {
        removed.add(index);
        const message = messages[index];
        if (message) remainingTokens -= estimateMessageTokens(message);
      }
    }
    const output = messages.filter((_message, index) => !removed.has(index));
    const estimatedTokensAfter = cost(output);
    return {
      messages: output,
      estimatedTokensBefore,
      estimatedTokensAfter,
      trimmedMessages: removed.size,
      reason: estimatedTokensAfter > this.maxTokenEstimate ? "protected_context_exceeds_budget" : "oldest_unprotected_trimmed",
    };
  }
}

function estimateMessageTokens(message: AgentMessage): number {
  const serialized = JSON.stringify(message);
  return Math.ceil((serialized?.length ?? 0) / 4);
}

/**
 * Only remove old assistant messages and complete tool-call groups. User/system
 * messages, recent-window messages, orphan tool results and incomplete calls
 * are protected. This mirrors CodeFlow's requirement to preserve tool closure.
 */
function findRemovableGroups(messages: readonly AgentMessage[], recentStart: number): Array<{ start: number; end: number }> {
  const groups: Array<{ start: number; end: number }> = [];
  let index = 0;
  while (index < recentStart) {
    const message = messages[index];
    if (!message || message.role !== "assistant") {
      index++;
      continue;
    }

    const toolCallIds = getAssistantToolCallIds(message);
    if (toolCallIds.length === 0) {
      groups.push({ start: index, end: index });
      index++;
      continue;
    }

    const pending = new Set(toolCallIds);
    let end = index;
    for (let next = index + 1; next < messages.length && pending.size > 0; next++) {
      const candidate = messages[next];
      if (!candidate || candidate.role !== "toolResult") break;
      const callId = getToolResultCallId(candidate);
      if (callId) pending.delete(callId);
      end = next;
    }
    if (pending.size === 0 && end < recentStart) {
      groups.push({ start: index, end });
      index = end + 1;
    } else {
      // Do not prune part of an incomplete or recent tool-call group.
      index = Math.max(index + 1, end + 1);
    }
  }
  return groups;
}

function getAssistantToolCallIds(message: AgentMessage): string[] {
  const content = (message as { content?: unknown }).content;
  if (!Array.isArray(content)) return [];
  return content.flatMap((part: unknown) => {
    if (!part || typeof part !== "object") return [];
    const item = part as { type?: unknown; id?: unknown };
    return item.type === "toolCall" && typeof item.id === "string" ? [item.id] : [];
  });
}

function getToolResultCallId(message: AgentMessage): string | undefined {
  const callId = (message as { toolCallId?: unknown }).toolCallId;
  return typeof callId === "string" ? callId : undefined;
}
