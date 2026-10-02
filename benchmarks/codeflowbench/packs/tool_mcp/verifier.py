from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from benchmarks.codeflowbench.canonical import canonical_json
from benchmarks.codeflowbench.records import (
    VerificationState,
    VerifierResult,
)

from .models import ToolMCPTask


def mcp_verifier_code_digest() -> str:
    return sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class SealedMCPReceiptVerifier:
    task: ToolMCPTask
    expected_data_digest: str
    verifier_code_digest: str

    @classmethod
    def capture(cls, task: ToolMCPTask) -> SealedMCPReceiptVerifier:
        return cls(
            task=task,
            expected_data_digest=task.expected_receipts_digest,
            verifier_code_digest=mcp_verifier_code_digest(),
        )

    def verify(
        self,
        receipt_path: Path,
    ) -> tuple[VerifierResult, tuple[dict, ...]]:
        if self.expected_data_digest != self.task.expected_receipts_digest:
            return (
                VerifierResult(
                    state=VerificationState.NOT_RUN,
                    findings=("mcp_expected_data_digest_changed",),
                ),
                (),
            )
        if self.verifier_code_digest != mcp_verifier_code_digest():
            return (
                VerifierResult(
                    state=VerificationState.NOT_RUN,
                    findings=("mcp_verifier_code_digest_changed",),
                ),
                (),
            )
        if not receipt_path.is_file():
            return (
                VerifierResult(
                    state=VerificationState.FAILED,
                    findings=("missing_mcp_receipt_log",),
                ),
                (),
            )
        try:
            receipts = tuple(
                json.loads(line) for line in receipt_path.read_text(encoding="utf-8").splitlines() if line.strip()
            )
        except (OSError, json.JSONDecodeError) as exc:
            return (
                VerifierResult(
                    state=VerificationState.NOT_RUN,
                    findings=(f"mcp_receipt_verifier_error:{type(exc).__name__}",),
                ),
                (),
            )
        if not all(isinstance(receipt, dict) for receipt in receipts):
            return (
                VerifierResult(
                    state=VerificationState.FAILED,
                    findings=("invalid_mcp_receipt_shape",),
                ),
                receipts,
            )
        expected = Counter(canonical_json(target.expected_receipt) for target in self.task.targets)
        observed = Counter(canonical_json(receipt) for receipt in receipts)
        missing = tuple(encoded for encoded, count in expected.items() if observed[encoded] < count)
        if missing:
            return (
                VerifierResult(
                    state=VerificationState.FAILED,
                    findings=("expected_mcp_receipt_missing",),
                    metrics={
                        "expected_receipt_count": len(self.task.targets),
                        "observed_receipt_count": len(receipts),
                    },
                ),
                receipts,
            )
        unexpected = tuple(encoded for encoded, count in observed.items() if count > expected[encoded])
        if unexpected:
            return (
                VerifierResult(
                    state=VerificationState.FAILED,
                    findings=("unexpected_mcp_receipt",),
                    metrics={
                        "expected_receipt_count": len(self.task.targets),
                        "observed_receipt_count": len(receipts),
                    },
                ),
                receipts,
            )
        return (
            VerifierResult(
                state=VerificationState.PASSED,
                metrics={
                    "expected_receipt_count": len(self.task.targets),
                    "observed_receipt_count": len(receipts),
                },
            ),
            receipts,
        )


__all__ = [
    "SealedMCPReceiptVerifier",
    "mcp_verifier_code_digest",
]
