"""Cog predictor for OpenAI Privacy Filter (PII detection and redaction).

Uses HuggingFace transformers pipeline for token classification.
"""

from __future__ import annotations

from typing import Any

import torch
from cog import BasePredictor, Input
from transformers import pipeline

MODEL_ID = "openai/privacy-filter"


class Predictor(BasePredictor):
    def setup(self) -> None:
        device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline(
            task="token-classification",
            model=MODEL_ID,
            device=device,
        )

    def predict(
        self,
        text: str = Input(description="Text to scan for PII"),
        aggregation_strategy: str = Input(
            description="How to group token-level predictions into spans. 'simple' is recommended.",
            default="simple",
            choices=["none", "simple", "first", "average", "max"],
        ),
    ) -> dict[str, Any]:
        """Detect and redact PII in text.

        Returns:
            redacted_text: input text with detected PII replaced by [LABEL] placeholders
            spans: list of detected spans with label, text, score, start, end
            summary: counts by label
        """
        raw_spans = self.classifier(text, aggregation_strategy=aggregation_strategy)
        spans = [self._normalize_span(s) for s in raw_spans]
        redacted_text = self._redact(text, spans)
        summary = self._summarize(spans)
        return {
            "redacted_text": redacted_text,
            "spans": spans,
            "summary": summary,
        }

    def _normalize_span(self, span: dict[str, Any]) -> dict[str, Any]:
        label = span.get("entity_group") or span.get("entity") or "unknown"
        return {
            "label": str(label),
            "text": span.get("word", ""),
            "score": float(span.get("score", 0.0)),
            "start": int(span.get("start", 0)),
            "end": int(span.get("end", 0)),
        }

    def _redact(self, text: str, spans: list[dict[str, Any]]) -> str:
        valid_spans = [s for s in spans if s["end"] > s["start"]]
        valid_spans.sort(key=lambda s: (s["start"], s["end"]))
        parts: list[str] = []
        cursor = 0
        for span in valid_spans:
            start = max(cursor, span["start"])
            end = max(start, span["end"])
            if start > cursor:
                parts.append(text[cursor:start])
            parts.append(f"[{span['label'].upper()}]")
            cursor = end
        parts.append(text[cursor:])
        return "".join(parts)

    def _summarize(self, spans: list[dict[str, Any]]) -> dict[str, Any]:
        by_label: dict[str, int] = {}
        for span in spans:
            label = span["label"]
            by_label[label] = by_label.get(label, 0) + 1
        return {
            "span_count": len(spans),
            "by_label": dict(sorted(by_label.items())),
        }
