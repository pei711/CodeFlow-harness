"""Offline evidence invariants for the optional Langfuse adapter."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from langfuse_codeflow import load_spans, make_events, sanitize, send_historical_spans


class AdapterEvidenceTests(unittest.TestCase):
    def test_terminal_span_replaces_in_progress_checkpoint(self):
        rows = [
            {"spanId": "one", "startTime": "2026-11-01T00:00:00Z", "done": False},
            {"spanId": "one", "startTime": "2026-11-01T00:00:00Z", "done": True},
        ]
        with patch.object(Path, "read_text", return_value="\n".join(json.dumps(row) for row in rows)):
            self.assertEqual(load_spans(Path("spans.log")), [rows[-1]])

    def test_sensitive_fields_and_inline_keys_are_redacted(self):
        result = sanitize({"nested": [{"apiKey": "private"}], "output": "token sk-example123456"})
        self.assertEqual(result, {"nested": [{"apiKey": "[REDACTED]"}], "output": "token [REDACTED]"})

    def test_prompt_token_total_includes_cache_reads(self):
        result = dict(
            run_id="cached",
            task="test",
            task_id="test",
            commit="test",
            model="test",
            started_at="2026-11-01T00:00:00+00:00",
            finished_at="2026-11-01T00:00:02+00:00",
            task_latency_ms=2000,
            return_code=0,
            timed_out=False,
            verifier_passed=True,
        )
        span = dict(
            spanId="model",
            name="llm.call",
            startTime=result["started_at"],
            endTime=result["finished_at"],
            attributes={
                "span.type": "model",
                "llm.usage.input_tokens": 120,
                "llm.usage.cache_read_tokens": 80,
                "llm.usage.output_tokens": 30,
            },
        )
        _, events = make_events(result, [span])
        generation = next(e for e in events if e.type == "generation-create")
        self.assertEqual(generation.body.usage_details, {"input": 200, "output": 30})

    def test_historical_time_and_unknown_verifier_survive_conversion(self):
        from langfuse import Langfuse
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        from opentelemetry.sdk.trace.id_generator import IdGenerator

        result = dict(
            run_id="offline-invariant",
            task="test",
            task_id="test",
            commit="test",
            model="test",
            started_at="2026-11-01T00:00:00+00:00",
            finished_at="2026-11-01T00:00:02+00:00",
            task_latency_ms=2000,
            return_code=0,
            timed_out=False,
            verifier_passed=None,
            evidence_metadata={"experiment_version": "frozen-v1", "failure_category": "unknown"},
        )
        trace_id, events = make_events(result, [])
        self.assertEqual([e.body.name for e in events if e.type == "score-create"], ["task_latency_ms"])
        self.assertEqual(trace_id, make_events(result, [])[0])

        class Ids(IdGenerator):
            next_span_id = 1

            def generate_trace_id(self):
                return int(trace_id, 16)

            def generate_span_id(self):
                return self.next_span_id

        generator = Ids()
        exporter = InMemorySpanExporter()
        provider = TracerProvider(id_generator=generator)
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        # Disable network resources; public wrappers still encode the supplied OTel span.
        client = Langfuse(public_key="pk-offline", secret_key="sk-offline", tracing_enabled=False)
        try:
            send_historical_spans(
                client,
                provider.get_tracer("offline-test"),
                generator,
                events,
                experiment_attributes={"langfuse.experiment.id": "offline-experiment"},
            )
            spans = exporter.get_finished_spans()
            self.assertEqual(len(spans), 1)
            self.assertEqual(spans[0].end_time - spans[0].start_time, 2_000_000_000)
            self.assertEqual(spans[0].context.trace_id, int(trace_id, 16))
            self.assertEqual(spans[0].attributes["langfuse.experiment.id"], "offline-experiment")
            self.assertEqual(spans[0].attributes["langfuse.observation.metadata.experiment_version"], "frozen-v1")
            self.assertEqual(spans[0].attributes["langfuse.observation.metadata.failure_category"], "unknown")
        finally:
            client.shutdown()
            provider.shutdown()


if __name__ == "__main__":
    unittest.main()
