import test from "node:test";
import assert from "node:assert/strict";
import { ContextBudgetManager } from "../src/context.ts";

const toolCall = (id) => ({ type: "toolCall", id, name: "read", arguments: { path: "file.ts" } });
const user = (text) => ({ role: "user", content: [{ type: "text", text }] });
const assistant = (content) => ({ role: "assistant", content });
const toolResult = (toolCallId, text) => ({ role: "toolResult", toolCallId, content: [{ type: "text", text }] });
const recentMessages = () => Array.from({ length: 8 }, (_, i) => user(`recent-${i}`));

test("leaves a transcript untouched when it fits the estimated budget", () => {
  const messages = [user("hello"), assistant([{ type: "text", text: "world" }])];
  const result = new ContextBudgetManager(10_000).apply(messages);

  assert.equal(result.reason, "within_budget");
  assert.equal(result.trimmedMessages, 0);
  assert.deepEqual(result.messages, messages);
});

test("prunes a complete multi-call assistant/tool-result group atomically", () => {
  const group = [
    assistant([toolCall("call-1"), toolCall("call-2")]),
    toolResult("call-1", "one".repeat(80)),
    toolResult("call-2", "two".repeat(80)),
  ];
  const messages = [user("protected user prompt"), ...group, ...recentMessages()];
  const result = new ContextBudgetManager(1).apply(messages);

  assert.equal(result.reason, "protected_context_exceeds_budget");
  assert.equal(result.trimmedMessages, 3);
  assert.deepEqual(result.messages, [messages[0], ...messages.slice(-8)]);
  assert.equal(result.messages.some((message) => message.role === "toolResult"), false);
});

test("keeps incomplete tool-call groups and their orphan results", () => {
  const incomplete = assistant([toolCall("missing-result")]);
  const orphan = toolResult("unmatched-result", "orphan");
  const messages = [user("prompt"), incomplete, orphan, ...recentMessages()];
  const result = new ContextBudgetManager(1).apply(messages);

  assert.equal(result.messages.includes(incomplete), true);
  assert.equal(result.messages.includes(orphan), true);
});

test("preserves the recent window and protected messages even over budget", () => {
  const system = { role: "system", content: "system rules" };
  const oldAnswer = assistant([{ type: "text", text: "old answer" }]);
  const recent = recentMessages();
  const result = new ContextBudgetManager(1).apply([system, user("prompt"), oldAnswer, ...recent]);

  assert.equal(result.messages.includes(system), true);
  assert.equal(result.messages.some((message) => message.role === "user"), true);
  assert.equal(result.messages.includes(oldAnswer), false);
  assert.deepEqual(result.messages.slice(-8), recent);
  assert.equal(result.reason, "protected_context_exceeds_budget");
});
