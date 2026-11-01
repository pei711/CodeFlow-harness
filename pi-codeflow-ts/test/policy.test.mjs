import test from "node:test";
import assert from "node:assert/strict";
import { ToolPolicyRegistry } from "../src/policy.ts";

test("allows read tools by effect and denies writes when not enabled", () => {
  const policy = new ToolPolicyRegistry();

  assert.deepEqual(policy.decide("read", new Set(["read"])), { allowed: true });
  assert.deepEqual(policy.decide("bash", new Set(["read"])), {
    allowed: false,
    reason: "tool effect 'execute' is disabled for this turn",
  });
});

test("fails closed for undeclared tools and permits explicitly registered effects", () => {
  const policy = new ToolPolicyRegistry();

  assert.equal(policy.decide("custom_search", new Set(["read"])).allowed, false);
  policy.register("custom_search", "read");
  assert.equal(policy.decide("custom_search", new Set(["read"])).allowed, true);
  policy.register("publish", "external");
  assert.equal(policy.decide("publish", new Set(["read"])).allowed, false);
  assert.equal(policy.decide("publish", new Set(["external"])).allowed, true);
});

test("rejects empty tool names and unknown as an explicit capability", () => {
  const policy = new ToolPolicyRegistry();

  assert.throws(() => policy.register("  ", "read"), /toolName must not be empty/);
  assert.throws(() => policy.register("custom", "unknown"), /unknown is implicit/);
});
