"""In-session candidate evidence registry."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any


def canonical_expression(expr: str) -> str:
    return re.sub(r"\s+", "", expr or "")


@dataclass
class CandidateRecord:
    candidate_id: str
    expression: str
    factor_name: str
    state: str = "draft"
    evidences: list[dict[str, Any]] = field(default_factory=list)
    reviews: list[dict[str, Any]] = field(default_factory=list)


class CandidateRegistry:
    def __init__(self) -> None:
        self._records: dict[str, CandidateRecord] = {}

    def record_evaluation(self, evidence: dict[str, Any]) -> CandidateRecord:
        candidate = evidence.get("candidate") if isinstance(evidence.get("candidate"), dict) else {}
        expr = str(candidate.get("expression") or "")
        factor_name = str(candidate.get("factor_name") or "expr")
        digest = hashlib.sha256(canonical_expression(expr).encode("utf-8")).hexdigest()[:16]
        record = self._records.get(digest)
        if record is None:
            record = CandidateRecord(candidate_id=f"cand_{digest}", expression=expr, factor_name=factor_name)
            self._records[digest] = record
        row = {
            "profile_id": evidence.get("profile", {}).get("profile_id"),
            "profile_hash": evidence.get("profile_hash"),
            "split": evidence.get("split"),
            "passed": evidence.get("passed"),
            "rule_results": evidence.get("rule_results", []),
            "metrics": evidence.get("metrics", {}),
        }
        record.evidences.append(row)
        split = row["split"]
        if split == "train":
            record.state = "train_evaluated" if row["passed"] else "train_rejected"
        elif split == "val":
            record.state = "validation_evaluated" if row["passed"] else "validation_rejected"
        elif split == "full":
            record.state = "delivery_evaluated" if row["passed"] else "delivery_rejected"
        return record

    def record_review(self, candidate_id: str, review: dict[str, Any]) -> CandidateRecord | None:
        record = next((item for item in self._records.values() if item.candidate_id == candidate_id), None)
        if record is None:
            return None
        record.reviews.append(dict(review))
        verdict = review.get("verdict")
        if verdict == "approve":
            record.state = "reviewer_approved"
        elif verdict == "revise":
            record.state = "reviewer_revise"
        elif verdict == "reject":
            record.state = "reviewer_rejected"
        return record

    def get(self, candidate_id: str) -> CandidateRecord | None:
        return next((item for item in self._records.values() if item.candidate_id == candidate_id), None)
