from codeflow.evolver.compressor import Event, TrajectoryCompressor


def _tool_call(name: str, arguments: str) -> dict[str, object]:
    return {"function": {"name": name, "arguments": arguments}}


def test_compress_binds_tool_sections_to_current_assistant() -> None:
    output = TrajectoryCompressor().compress(
        [
            Event(
                event_type="assistant",
                content="first turn",
                tool_calls=[_tool_call("first_tool", '{"path":"first"}')],
            ),
            Event(event_type="tool", content="first result"),
            Event(
                event_type="assistant",
                content="second turn",
                tool_calls=[_tool_call("second_tool", '{"path":"second"}')],
            ),
            Event(event_type="tool", content="second result"),
        ]
    )

    assert output == (
        "=== TRAJECTORY SUMMARY ===\n"
        "Total turns: 2 | Empty-content turns: 0 | Anomalies flagged: 0\n"
        "\n--- TURNS ---\n"
        "\nTurn 1 (assistant):\n"
        "  first turn\n"
        '  → call first_tool({"path":"first"})\n'
        "  ← result: first result\n"
        "\nTurn 2 (assistant):\n"
        "  second turn\n"
        '  → call second_tool({"path":"second"})\n'
        "  ← result: second result"
    )


def test_compress_skips_stray_tool_results_after_assistant_without_calls() -> None:
    output = TrajectoryCompressor().compress(
        [
            Event(event_type="assistant", content="no tool call"),
            Event(event_type="tool", content="stray result"),
            Event(event_type="assistant", content="next turn"),
        ]
    )

    assert output == (
        "=== TRAJECTORY SUMMARY ===\n"
        "Total turns: 2 | Empty-content turns: 0 | Anomalies flagged: 0\n"
        "\n--- TURNS ---\n"
        "\nTurn 1 (assistant):\n"
        "  no tool call\n"
        "\nTurn 2 (assistant):\n"
        "  next turn"
    )
