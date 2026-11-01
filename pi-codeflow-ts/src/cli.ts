import { PiTurnRunner } from "./runner.js";

const [workspace, ...promptParts] = process.argv.slice(2);
if (!workspace || promptParts.length === 0) {
  console.error("Usage: npm start -- <workspace> <prompt...>");
  process.exitCode = 2;
} else {
  const runner = new PiTurnRunner();
  try {
    const outcome = await runner.run(
      {
        turnId: crypto.randomUUID(),
        conversationId: "cli:direct",
        workspace,
        origin: "user",
        text: promptParts.join(" "),
      },
      async (event) => {
        if (event.type === "text_delta") process.stdout.write(event.delta);
        else if (event.type === "notice") process.stderr.write(`\n${event.message}\n`);
      },
      () => [],
    );
    process.stdout.write(`\n[${outcome.status}; tools=${outcome.toolCalls}; failures=${outcome.toolFailures}]\n`);
    if (outcome.status !== "completed") process.exitCode = 1;
  } finally {
    await runner.closeConversation("cli:direct");
  }
}
