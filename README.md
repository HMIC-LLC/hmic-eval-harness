# HMIC Eval Harness

A reusable, small benchmark runner for **tool calls and grounded final answers**. It attaches to two explicitly selected OpenAI-compatible endpoints, replays the same synthetic tool results, preserves the returned responses, and produces a case-by-case comparison. You can replace the included YAML suite with your own. No EdgeProof package, GGUF path, GPU launcher, or particular model is required.

This repository includes **HMIC Cross-Skill 12** as a diagnostic example. Those twelve cases grew from HMIC's local model evaluation work and from a companion to [EdgeProof](https://github.com/Techn08055/edgeproof). The [EdgeProof-specific companion](https://github.com/HMIC-LLC/edgeproof-cross-skill-12) remains available for that author's F16/Q4 experiment. This harness is the reusable part.

## What it measures

The example covers exact output format, basic route control, grounded answers when only a mock result changes, an injected instruction inside a tool result, multi-call dependencies, timeout recovery, permanent errors, and a superseded notice. It checks both the tool turn and the final answer. Unauthorized extra calls are failures even when the final text would have been right.

The fixtures are synthetic. The runner **never executes** the advertised `ticket_update`, `file_size`, or other suite tools. It sends the predetermined `mock_result` or `mock_results` only after the assistant makes the expected call(s). A failed tool turn stops that case; its hypothetical final answer is not measured.

The twelve cases are selected diagnostics, **not a statistically powered benchmark**. Repeats of one prompt are not independent tasks. A result from two different models, server builds, chat templates, placements, or inference settings is a configuration comparison; do not attribute the difference to quantization alone.

## Install and validate without inference

Python 3.10+ is required. Commands below use PowerShell. Install dependencies in your preferred environment:

```powershell
python -m pip install --break-system-packages -r '.\requirements.txt'
python -B '.\bench.py' validate --suite '.\examples\cross-skill-12\suite.yaml'
python -B -m unittest discover -s '.\tests' -p 'test_*.py' -v
```

`validate` checks suite structure, unique IDs, JSON schemas, expected tool references, and mock-result counts. Tests use scripted responses and a loopback HTTP server. Neither starts a model.

## Run a comparison

Both endpoints must already be serving the named model. Give the **base URL ending in `/v1`**, not the full chat path. This program is attach-only: it does not load, launch, switch, or download a model. It first checks `/v1/models` for the requested model ID, then calls `/v1/chat/completions`. Some servers omit `/models`; use `--skip-model-preflight` when needed. A response that names a different model is recorded as an identity error.

```powershell
python -B '.\bench.py' run `
  --suite '.\examples\cross-skill-12\suite.yaml' `
  --reference-url 'http://127.0.0.1:8080/v1' --reference-model 'model-f16' `
  --candidate-url 'http://127.0.0.1:8081/v1' --candidate-model 'model-q4' `
  --out 'C:\bench\hmic-cross-skill-12-run-01'
```

If an endpoint requires authentication, set an environment variable locally and pass **its name**, not its value, with `--reference-key-env` or `--candidate-key-env`. The variable value is sent as a Bearer token and is not put into the saved request settings. Do not put credentials in the endpoint URL. Inspect `report.json` before sharing it: raw model replies and prompts can contain sensitive text if you use a private suite.

Use `--repeats N` to override suite repeats. Seed is sent as 0..N-1 by default; `--omit-seed` is available for endpoints that reject it. Likewise, `--omit-parallel-tool-calls` omits that optional request key. Whether a server actually honors these settings is a separate compatibility question; the harness records what it requested and the raw response, not an unverified effective setting. `--timeout` sets seconds per HTTP request.

The output directory must be new. `report.json` stores full returned response objects, parsed tool calls, mock results fed, final answer judgments, finish reasons, and errors. `comparison.md` is a compact case table. A completed reference side is saved before the candidate side runs, so a candidate failure leaves a partial receipt.

## Author a suite

Copy `examples/cross-skill-12/suite.yaml` and change the name, tools, tasks, and expected results. The suite may contain any positive number of tasks. Each task has a unique `id`, a `prompt`, one or more `steps`, and a final `response`. Each step expects zero calls, one `{tool, args}` call, or a `calls` list. For every expected call, provide one synthetic result. `order: strict` is available when call order matters; otherwise calls are matched by tool name and exact argument object. `response.type` is `exact` (byte exact output, with separate content/format judgments) or `json` (structural JSON equality).

The runner sends the suite's tool JSON schemas to the endpoint and validates returned arguments. It labels missing, extra, malformed, unauthorized side-effect, premature, schema-invalid, and truncated calls separately. It does not infer intent from a string that the server failed to expose as a tool call; inspect raw responses before deciding whether the model or the server/parser failed.

## Offline regrade

```powershell
python -B '.\bench.py' regrade `
  --suite '.\examples\cross-skill-12\suite.yaml' `
  --report 'C:\bench\hmic-cross-skill-12-run-01\report.json' `
  --out 'C:\bench\hmic-cross-skill-12-run-01\regraded.json'
```

Regrade does not call a model. It requires the original suite file's SHA-256 and refuses to overwrite the original report. The original raw responses remain in the report. If the endpoint omitted a tool call from its returned structured response, regrading cannot reconstruct it from invisible internal output.

## Comparison discipline

For a quantization comparison, pin the underlying model and prompt, both weight artifacts and hashes, tokenizer/chat template, server build, context and output limits, sampling settings, tool schema, and CPU/GPU placement. If the full-precision variant spills to CPU while Q4 fits in GPU memory, call it a **system configuration comparison**. Include any route-control failure before interpreting later tool cases. Review individual disagreements and raw responses before making a model-level claim.

MIT licensed. HMIC LLC, 2026. No model results are bundled; the included suite has only synthetic fixtures.
