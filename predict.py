"""Cog predictor for OpenAI Privacy Filter (PII detection and redaction)."""

from cog import BasePredictor, Input

from opf._api import OPF, RedactionResult


class Predictor(BasePredictor):
    def setup(self):
        """Load the model onto GPU. Weights are pre-downloaded at build time."""
        self.model = OPF(device="cuda", output_mode="typed")
        # Warm up: triggers model load and JIT compilation
        self.model.redact("warmup")

    def predict(
        self,
        text: str = Input(description="Text to scan for PII"),
        output_mode: str = Input(
            description="'typed' preserves category labels like PRIVATE_PERSON, 'redacted' collapses all to a single REDACTED label",
            choices=["typed", "redacted"],
            default="typed",
        ),
        decode_mode: str = Input(
            description="'viterbi' for coherent span boundaries (recommended), 'argmax' for raw per-token prediction",
            choices=["viterbi", "argmax"],
            default="viterbi",
        ),
    ) -> dict:
        """Detect and redact PII in text.

        Returns a structured result with the original text, detected spans
        (with labels, positions, and text), and the redacted text with
        placeholders like [PRIVATE_PERSON], [PRIVATE_EMAIL], etc.
        """
        self.model.set_output_mode(output_mode)
        self.model.set_decode_mode(decode_mode)
        result = self.model.redact(text)
        if isinstance(result, RedactionResult):
            return result.to_dict()
        return {"redacted_text": result}
