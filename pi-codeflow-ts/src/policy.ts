/** Classify Pi tools with the effect labels used by CodeFlow's tool registry. */

export type ToolEffect = "read" | "write" | "execute" | "external" | "unknown";

export interface ToolDecision {
  readonly allowed: boolean;
  readonly reason?: string;
}

export class ToolPolicyRegistry {
  readonly #effects = new Map<string, ToolEffect>();

  constructor() {
    this.#effects.set("read", "read");
    this.#effects.set("write", "write");
    this.#effects.set("edit", "write");
    this.#effects.set("bash", "execute");
  }

  register(toolName: string, effect: ToolEffect): void {
    if (!toolName.trim()) throw new Error("toolName must not be empty");
    if (effect === "unknown") throw new Error("unknown is implicit; register a concrete tool effect");
    this.#effects.set(toolName, effect);
  }

  decide(toolName: string, allowedEffects: ReadonlySet<ToolEffect>): ToolDecision {
    const effect = this.#effects.get(toolName) ?? "unknown";
    if (effect === "unknown") return { allowed: false, reason: `tool '${toolName}' has no declared CodeFlow effect` };
    if (!allowedEffects.has(effect)) return { allowed: false, reason: `tool effect '${effect}' is disabled for this turn` };
    return { allowed: true };
  }
}
