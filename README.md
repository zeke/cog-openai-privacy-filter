# cog-openai-privacy-filter

Cog packaging of [OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter) for deployment on Replicate.

- Replicate model: https://replicate.com/zeke/openai-privacy-filter
- HuggingFace weights: https://huggingface.co/openai/privacy-filter
- OpenAI announcement: https://openai.com/index/introducing-openai-privacy-filter/
- Upstream code: https://github.com/openai/privacy-filter
- License: Apache 2.0

## What it does

OpenAI Privacy Filter is a 1.5B parameter token-classification model (50M active via mixture-of-experts) that detects personally identifiable information (PII) in unstructured text. It produces structured spans with labels and confidence scores, plus a redacted version of the input text with placeholders.

The model detects 8 categories of PII:

| Label             | Examples                                |
| ----------------- | --------------------------------------- |
| `private_person`  | Names of private individuals            |
| `private_email`   | Email addresses                         |
| `private_phone`   | Phone numbers                           |
| `private_address` | Postal addresses                        |
| `private_date`    | Dates of birth, appointment dates, etc. |
| `private_url`     | URLs that identify individuals          |
| `account_number`  | Bank accounts, credit cards, IDs        |
| `secret`          | API keys, passwords, tokens             |

## Example

```bash
curl -s -X POST \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "text": "My name is Alice Smith and my email is alice@example.com. My API key is sk-abc123."
    }
  }' \
  https://api.replicate.com/v1/models/zeke/openai-privacy-filter/predictions
```

Output:

```json
{
  "redacted_text": "My name is [PRIVATE_PERSON] and my email is [PRIVATE_EMAIL]. My API key is [SECRET].",
  "spans": [
    { "label": "private_person", "text": "Alice Smith", "score": 0.999, "start": 11, "end": 22 },
    { "label": "private_email", "text": "alice@example.com", "score": 0.999, "start": 39, "end": 56 },
    { "label": "secret", "text": "sk-abc123", "score": 0.999, "start": 73, "end": 82 }
  ],
  "summary": {
    "span_count": 3,
    "by_label": { "private_email": 1, "private_person": 1, "secret": 1 }
  }
}
```

## Inputs

| Input                  | Type   | Default    | Description                                                            |
| ---------------------- | ------ | ---------- | ---------------------------------------------------------------------- |
| `text`                 | string | (required) | Text to scan for PII                                                   |
| `aggregation_strategy` | enum   | `simple`   | How to group token-level predictions: `none`, `simple`, `first`, `average`, `max` |

## Outputs

| Field           | Type    | Description                                                          |
| --------------- | ------- | -------------------------------------------------------------------- |
| `redacted_text` | string  | Input text with detected PII replaced by `[LABEL]` placeholders      |
| `spans`         | array   | Detected spans, each with `label`, `text`, `score`, `start`, `end`   |
| `summary`       | object  | Aggregate stats: `span_count` and `by_label` counts                  |

## How the Cog model works

This is a thin wrapper around HuggingFace's `transformers.pipeline` for token classification.

1. `setup()` loads the `openai/privacy-filter` checkpoint via `transformers.pipeline(task="token-classification")`. This downloads the model from HuggingFace Hub on the first cold boot (~2.8 GB) and pins it to GPU if available. Subsequent predictions on a warm container reuse the loaded model.
2. `predict()` runs the input text through the pipeline with the chosen `aggregation_strategy`. The pipeline returns token-level predictions grouped into spans.
3. Adjacent spans with the same label are merged so multi-token entities (like full names or long emails) come back as a single span.
4. Span boundaries are trimmed of leading and trailing whitespace, since the tokenizer often includes a leading space in subword tokens.
5. The redacted text is built by replacing each span's character range with `[LABEL]`.

The Cog config (`cog.yaml`) declares Python 3.12, torch 2.6 with CUDA 12.6, transformers, and accelerate. Replicate runs the model on a T4 GPU.

## Running locally

```bash
git clone https://github.com/zeke/cog-openai-privacy-filter
cd cog-openai-privacy-filter
cog predict -i text="Alice emailed bob@example.com from +1 (415) 555-0124"
```

To push a new version to Replicate:

```bash
cog push r8.im/zeke/openai-privacy-filter
```

## Addendum: why not the official `opf` package?

OpenAI ships an official inference package, [`opf`](https://github.com/openai/privacy-filter), that loads the same weights with a custom Viterbi CRF decoder for more coherent span boundaries. I tried that path first.

It didn't work out of the box on Replicate. The `opf` setup consistently failed with:

```
worker setup failed: setup failed: IndexError: map::at
```

That error comes from a C++/Rust extension, not Python. The most likely culprit is `tiktoken`, which `opf` uses for tokenization. tiktoken's Rust BPE backend lazily downloads BPE merge files at runtime (e.g. `o200k_base.tiktoken` from `openaipublic.blob.core.windows.net`). When that download fails or returns malformed data, the Rust code throws `map::at`. The Replicate prediction container couldn't reach that URL cleanly, and even pre-caching the BPE file at build time didn't resolve it.

I switched to HuggingFace `transformers`, which:

- Uses the bundled `tokenizer.json` file from the HF repo instead of tiktoken
- Loads weights via standard `AutoModelForTokenClassification` instead of `opf`'s custom MXFP4 unpacking and MoE sharding
- Has well-tested model caching, GPU dispatch, and pipeline aggregation

The trade-off is losing the Viterbi CRF decoder. In practice the model still produces high-confidence, accurate spans (>99% on simple tests), and merging adjacent same-label spans recovers most of the span-coherence benefit.
