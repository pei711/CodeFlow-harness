import test from "node:test";
import assert from "node:assert/strict";
import { createAgentSession, SessionManager } from "@earendil-works/pi-coding-agent";

test("Pi 1.4.0 creates and disposes an in-memory session for the runner", async () => {
  const { session } = await createAgentSession({
    cwd: process.cwd(),
    sessionManager: SessionManager.inMemory(),
    noTools: true,
  });

  try {
    assert.equal(session.isStreaming, false);
    assert.equal(typeof session.agent.subscribe, "function");
    assert.equal(typeof session.prompt, "function");
  } finally {
    session.dispose();
  }
});
