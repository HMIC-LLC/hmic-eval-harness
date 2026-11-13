"""HMIC reusable synthetic tool-loop benchmark. Python 3.10+."""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import platform
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import jsonschema
import yaml


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def load_suite(path):
    raw = Path(path).read_bytes()
    suite = yaml.safe_load(raw)
    _need(isinstance(suite, dict), "suite must be an object")
    _need(isinstance(suite.get("name"), str) and suite["name"], "suite needs a name")
    tools = suite.get("tools", [])
    tasks = suite.get("tasks")
    _need(isinstance(tools, list) and isinstance(tasks, list) and tasks, "suite needs tools list and nonempty tasks list")
    names = []
    for tool in tools:
        _need(isinstance(tool, dict) and isinstance(tool.get("name"), str), "tool needs name")
        _need(isinstance(tool.get("parameters"), dict), f"tool {tool['name']} needs JSON schema parameters")
        jsonschema.Draft202012Validator.check_schema(tool["parameters"])
        names.append(tool["name"])
    _need(len(names) == len(set(names)), "duplicate tool name")
    ids = []
    for task in tasks:
        _need(isinstance(task, dict) and isinstance(task.get("id"), str) and task["id"], "task needs id")
        ids.append(task["id"])
        _need(isinstance(task.get("prompt"), str) and task["prompt"], f"{task['id']}: prompt required")
        steps = task.get("steps")
        _need(isinstance(steps, list) and steps, f"{task['id']}: nonempty steps required")
        response = task.get("response")
        _need(isinstance(response, dict) and response.get("type") in ("exact", "json") and "value" in response,
              f"{task['id']}: response needs exact or json value")
        for step in steps:
            _need(isinstance(step, dict) and isinstance(step.get("expect"), dict), f"{task['id']}: step needs expect")
            expect = step["expect"]
            if "calls" in expect:
                calls = expect["calls"]
            elif expect.get("tool") is None:
                calls = []
            else:
                calls = [{k: v for k, v in expect.items() if k in ("tool", "args")}]
            _need(isinstance(calls, list), f"{task['id']}: calls must be list")
            for call in calls:
                _need(isinstance(call, dict) and call.get("tool") in names and isinstance(call.get("args", {}), dict),
                      f"{task['id']}: invalid expected tool call")
            _need(expect.get("order", "any") in ("any", "strict"), f"{task['id']}: invalid order")
            order = expect.get("order", "any")
            expect.clear()
            expect.update({"calls": calls, "order": order})
            if calls:
                results = step.get("mock_results", [step.get("mock_result")])
                _need(isinstance(results, list) and len(results) == len(calls) and all(r is not None for r in results),
                      f"{task['id']}: one mock result per expected call required")
                step["mock_results"] = results
            else:
                _need("mock_results" not in step and "mock_result" not in step,
                      f"{task['id']}: answer-only step cannot have mock results")
    _need(len(ids) == len(set(ids)), "duplicate task id")
    _need(isinstance(suite.get("repeats", 1), int) and suite.get("repeats", 1) > 0, "repeats must be positive")
    suite.setdefault("system", "")
    suite.setdefault("temperature", 0)
    suite.setdefault("repeats", 1)
    suite.setdefault("parallel_tool_calls", True)
    suite["sha256"] = hashlib.sha256(raw).hexdigest()
    return suite


def openai_tools(suite):
    return [{"type": "function", "function": {k: t[k] for k in ("name", "description", "parameters")}}
            for t in suite["tools"]]


def parse_calls(message):
    calls = []
    error = None
    for i, item in enumerate(message.get("tool_calls") or []):
        function = item.get("function") or {}
        name = function.get("name")
        raw = function.get("arguments", "")
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(args, dict):
                raise ValueError("arguments must decode to object")
        except (ValueError, TypeError):
            args = None
            error = "invalid_json_args"
        calls.append({"id": item.get("id") or f"call_{i}", "tool": name, "args": args,
                      "raw_arguments": raw})
    return calls, error


def score_calls(expect, calls, tools, later=(), parse_error=None):
    want = expect["calls"]
    failures = []
    if parse_error:
        failures.append(parse_error)
    schemas = {t["name"]: t["parameters"] for t in tools}
    side = {t["name"] for t in tools if t.get("side_effect")}
    used = set()
    matched = []
    for desired in want:
        hits = [(i, c) for i, c in enumerate(calls) if i not in used and c["tool"] == desired["tool"]]
        if not hits:
            failures.append("missing_call")
            matched.append(None)
            continue
        exact = [(i, c) for i, c in hits if c["args"] == desired.get("args", {})]
        i, c = (exact or hits)[0]
        used.add(i)
        matched.append(i)
        if c["args"] != desired.get("args", {}):
            failures.append("bad_args")
    for c in calls:
        if c["args"] is None:
            failures.append("invalid_json_args")
        elif c["tool"] not in schemas:
            failures.append("unknown_tool")
        else:
            try:
                jsonschema.validate(c["args"], schemas[c["tool"]])
            except jsonschema.ValidationError:
                failures.append("schema_error")
    extra = [c for i, c in enumerate(calls) if i not in used]
    if extra:
        failures.append("extra_call")
    if any(c["tool"] in side for c in extra):
        failures.append("unrequested_action")
    if any(c["tool"] in later for c in extra):
        failures.append("premature_action")
    if expect.get("order") == "strict" and [i for i in matched if i is not None] != sorted(i for i in matched if i is not None):
        failures.append("wrong_order")
    if want and not calls:
        failures.append("no_call_when_required")
    if not want and calls:
        failures.append("call_when_not_required")
    return {"passed": not failures, "failures": list(dict.fromkeys(failures)),
            "want_calls": want, "got_calls": calls, "assignment": matched,
            "unrequested_action": any(c["tool"] in side for c in extra)}


def score_response(spec, text, calls):
    if spec["type"] == "exact":
        content = text.strip() == spec["value"]
        fmt = text == spec["value"]
    else:
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            parsed = None
        content = parsed == spec["value"] and type(parsed) is type(spec["value"])
        fmt = content
    return {"content_ok": content, "format_ok": fmt, "no_tool_call": not calls,
            "passed": content and fmt and not calls}


class EndpointError(Exception):
    def __init__(self, code, detail=""):
        super().__init__(detail)
        self.code = code


class Endpoint:
    """Attach only. Never launches, loads, or changes a model."""
    def __init__(self, url, model, key_env=None, timeout=60, preflight=True):
        parsed = urllib.parse.urlparse(url)
        if (parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("endpoint URL must be absolute HTTP(S) without embedded credentials")
        self.url = url.rstrip("/")
        self.model = model
        self.key_env = key_env
        self.timeout = timeout
        self.preflight_enabled = preflight

    def _request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if self.key_env:
            key = os.environ.get(self.key_env)
            if not key:
                raise EndpointError("missing_api_key", f"environment variable {self.key_env} is unset")
            headers["Authorization"] = "Bearer " + key
        request = urllib.request.Request(self.url + path, data=data, headers=headers,
                                         method="POST" if data is not None else "GET")
        start = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read(16 * 1024 * 1024)
        except urllib.error.HTTPError as exc:
            # Do not print remote bodies: some servers echo Authorization headers.
            raise EndpointError("http_error", f"HTTP {exc.code}") from None
        except TimeoutError:
            raise EndpointError("timeout", "request timed out") from None
        except (urllib.error.URLError, OSError) as exc:
            raise EndpointError("transport_error", type(exc).__name__) from None
        try:
            decoded = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            raise EndpointError("invalid_response_json", "endpoint returned non-JSON response") from None
        return decoded, round((time.monotonic() - start) * 1000, 1)

    def preflight(self):
        if not self.preflight_enabled:
            return {"skipped": True}
        data, latency = self._request("/models")
        ids = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
        if self.model not in ids:
            raise EndpointError("model_not_listed", "requested model is absent from /models")
        return {"listed_model": self.model, "latency_ms": latency}

    def chat(self, messages, tools, settings):
        payload = {"model": self.model, "messages": messages,
                   "temperature": settings["temperature"]}
        if tools:
            payload["tools"] = tools
            if settings.get("parallel_tool_calls") is not None:
                payload["parallel_tool_calls"] = settings["parallel_tool_calls"]
        if settings.get("seed") is not None:
            payload["seed"] = settings["seed"]
        data, latency = self._request("/chat/completions", payload)
        if data.get("model") and data["model"] != self.model:
            raise EndpointError("model_identity_mismatch", "response model differs from requested model")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0].get("message"), dict):
            raise EndpointError("invalid_response_shape", "missing assistant message")
        return data, latency


def run_case(endpoint, suite, task, seed, settings):
    messages = []
    if suite["system"]:
        messages.append({"role": "system", "content": suite["system"]})
    messages.append({"role": "user", "content": task["prompt"]})
    tool_names = set(task.get("tools") or [t["name"] for t in suite["tools"]])
    tools = [t for t in openai_tools(suite) if t["function"]["name"] in tool_names]
    record = {"seed": seed, "turns": [], "passed": False}
    expected_turns = len(task["steps"]) + (1 if task["steps"][-1]["expect"]["calls"] else 0)
    for turn_no in range(expected_turns):
        try:
            raw, latency = endpoint.chat(messages, tools, {**settings, "seed": seed})
        except EndpointError as exc:
            record["turns"].append({"error": exc.code, "detail": str(exc), "passed": False})
            break
        choice = raw["choices"][0]
        message = choice["message"]
        calls, parse_error = parse_calls(message)
        text = message.get("content") or ""
        turn = {"raw_response": raw, "text": text, "calls": calls,
                "latency_ms": latency, "finish_reason": choice.get("finish_reason")}
        is_final = turn_no == len(task["steps"]) and expected_turns > len(task["steps"])
        if is_final:
            turn["score"] = score_response(task["response"], text, calls)
            if parse_error:
                turn["score"]["passed"] = False
                turn["score"]["error"] = parse_error
        else:
            step = task["steps"][turn_no]
            later = [c["tool"] for st in task["steps"][turn_no + 1:] for c in st["expect"]["calls"]]
            turn["score"] = score_calls(step["expect"], calls, suite["tools"], later, parse_error)
            if not step["expect"]["calls"]:
                turn["final_score"] = score_response(task["response"], text, calls)
        if choice.get("finish_reason") == "length":
            turn["score"]["passed"] = False
            turn["score"].setdefault("failures", []).append("truncated_completion")
        record["turns"].append(turn)
        if not turn["score"]["passed"]:
            break
        if not is_final and turn_no < len(task["steps"]):
            step = task["steps"][turn_no]
            if not step["expect"]["calls"]:
                break
            message_copy = {"role": "assistant", "content": message.get("content"),
                            "tool_calls": message.get("tool_calls") or []}
            messages.append(message_copy)
            fed = []
            assignments = sorted(enumerate(turn["score"]["assignment"]), key=lambda pair: pair[1])
            for want_idx, call_idx in assignments:
                call = calls[call_idx]
                result = step["mock_results"][want_idx]
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": json.dumps(result, separators=(",", ":"))})
                fed.append({"tool_call_id": call["id"], "result": result})
            turn["fed_results"] = fed
    record["passed"] = (len(record["turns"]) == expected_turns
                        and all(t.get("score", {}).get("passed", False) for t in record["turns"])
                        and record["turns"][-1].get("final_score", record["turns"][-1].get("score", {})).get("passed", False))
    return record


def regrade(report, suite):
    if report.get("suite_sha256") != suite["sha256"]:
        raise ValueError("suite hash mismatch; frozen gold changed")
    specs = {t["id"]: t for t in suite["tasks"]}
    for model in report["models"]:
        for item in model["tasks"]:
            task = specs[item["id"]]
            for run in item["runs"]:
                for i, turn in enumerate(run["turns"]):
                    if "error" in turn:
                        continue
                    calls = turn["calls"]
                    if i < len(task["steps"]):
                        step = task["steps"][i]
                        later = [c["tool"] for st in task["steps"][i + 1:] for c in st["expect"]["calls"]]
                        turn["score"] = score_calls(step["expect"], calls, suite["tools"], later)
                        if not step["expect"]["calls"]:
                            turn["final_score"] = score_response(task["response"], turn["text"], calls)
                    else:
                        turn["score"] = score_response(task["response"], turn["text"], calls)
                    if turn.get("finish_reason") == "length":
                        turn["score"]["passed"] = False
                        turn["score"].setdefault("failures", []).append("truncated_completion")
                expected = len(task["steps"]) + (1 if task["steps"][-1]["expect"]["calls"] else 0)
                run["passed"] = (len(run["turns"]) == expected
                                 and all(t.get("score", {}).get("passed", False) for t in run["turns"])
                                 and run["turns"][-1].get("final_score", run["turns"][-1].get("score", {})).get("passed", False))
    return report


def comparison(report):
    lines = [f"# {report['suite_name']} comparison", "", f"Suite SHA-256: `{report['suite_sha256']}`", ""]
    for model in report["models"]:
        total = sum(len(t["runs"]) for t in model["tasks"])
        passed = sum(r["passed"] for t in model["tasks"] for r in t["runs"])
        lines.append(f"- `{model['label']}`: {passed}/{total} end-to-end case runs")
    lines += ["", "| Case | " + " | ".join(m["label"] for m in report["models"]) + " |", "| --- | " + " | ".join("---:" for _ in report["models"]) + " |"]
    for i, task in enumerate(report["models"][0]["tasks"]):
        values = []
        for model in report["models"]:
            other = model["tasks"][i]
            _need(other["id"] == task["id"], "report task order mismatch")
            values.append(f"{sum(r['passed'] for r in other['runs'])}/{len(other['runs'])}")
        lines.append("| " + task["id"] + " | " + " | ".join(values) + " |")
    route_failures = []
    for model in report["models"]:
        if any(t["category"] == "route" and not all(r["passed"] for r in t["runs"])
               for t in model["tasks"]):
            route_failures.append(model["label"])
    if route_failures:
        lines += ["", "**Route control failed for: " + ", ".join(route_failures)
                  + ". Hold interpretation of later tool cases until that path is checked.**"]
    lines += ["", "Selected diagnostic cases do not establish an aggregate model or quantization effect.",
              "Repeated runs of the same prompt are not independent cases. Inspect `report.json` for raw replies,"
              "parsed calls, mock results, failures, and endpoint settings. A tool-route control failure makes"
              "later tool-case interpretation suspect. Endpoint compatibility is a measured part of the result.", ""]
    return "\n".join(lines)


def _exclusive_write(path, data):
    with Path(path).open("x", encoding="utf-8") as file:
        file.write(data)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("validate", "run", "regrade"))
    p.add_argument("--suite", required=True)
    p.add_argument("--out")
    p.add_argument("--report")
    p.add_argument("--reference-url")
    p.add_argument("--reference-model")
    p.add_argument("--reference-key-env")
    p.add_argument("--candidate-url")
    p.add_argument("--candidate-model")
    p.add_argument("--candidate-key-env")
    p.add_argument("--timeout", type=float, default=60)
    p.add_argument("--repeats", type=int)
    p.add_argument("--omit-seed", action="store_true")
    p.add_argument("--omit-parallel-tool-calls", action="store_true")
    p.add_argument("--skip-model-preflight", action="store_true")
    a = p.parse_args(argv)
    suite = load_suite(a.suite)
    if a.command == "validate":
        print(f"VALID {suite['name']}: {len(suite['tasks'])} cases; SHA-256 {suite['sha256']}")
        return 0
    if a.command == "regrade":
        _need(a.report and a.out, "regrade needs --report and distinct --out")
        _need(Path(a.report).resolve() != Path(a.out).resolve(), "refusing to overwrite original report")
        report = json.loads(Path(a.report).read_text(encoding="utf-8"))
        regrade(report, suite)
        _exclusive_write(a.out, json.dumps(report, indent=2, ensure_ascii=False))
        print(f"Regraded {a.out}")
        return 0
    _need(a.out and a.reference_url and a.reference_model and a.candidate_url and a.candidate_model,
          "run needs --out and reference/candidate URL + model")
    repeats = a.repeats or suite["repeats"]
    _need(repeats > 0, "repeats must be positive")
    endpoints = [Endpoint(a.reference_url, a.reference_model, a.reference_key_env, a.timeout, not a.skip_model_preflight),
                 Endpoint(a.candidate_url, a.candidate_model, a.candidate_key_env, a.timeout, not a.skip_model_preflight)]
    output = Path(a.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    _need(not report_path.exists() and not (output / "comparison.md").exists(), "output already exists")
    settings = {"temperature": suite["temperature"],
                "parallel_tool_calls": None if a.omit_parallel_tool_calls else suite["parallel_tool_calls"]}
    report = {"schema": "hmic-eval-harness-v1", "suite_name": suite["name"],
              "suite_sha256": suite["sha256"], "recorded_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "settings": {**settings, "repeats": repeats, "seed_sent": not a.omit_seed,
                           "timeout_s": a.timeout, "python": platform.python_version()}, "models": []}
    for label, endpoint in zip(("reference", "candidate"), endpoints):
        model = {"label": label, "endpoint_url": endpoint.url, "requested_model": endpoint.model, "tasks": []}
        try:
            model["preflight"] = endpoint.preflight()
        except EndpointError as exc:
            model["preflight"] = {"error": exc.code, "detail": str(exc)}
            report["models"].append(model)
            _exclusive_write(report_path, json.dumps(report, indent=2, ensure_ascii=False))
            raise
        for task in suite["tasks"]:
            runs = [run_case(endpoint, suite, task, None if a.omit_seed else seed, settings)
                    for seed in range(repeats)]
            model["tasks"].append({"id": task["id"], "category": task.get("category", "general"), "runs": runs})
            print(f"{label} {task['id']}: {sum(r['passed'] for r in runs)}/{repeats}", flush=True)
        report["models"].append(model)
        # Save after each model so a failed second side preserves the first.
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _exclusive_write(output / "comparison.md", comparison(report))
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, EndpointError) as exc:
        print(f"{type(exc).__name__}: {getattr(exc, 'code', '')} {exc}", file=sys.stderr)
        raise SystemExit(2)
