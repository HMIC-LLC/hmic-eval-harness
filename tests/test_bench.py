import copy
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "examples" / "cross-skill-12" / "suite.yaml"


def reply(text="", calls=(), model="m"):
    return {"model": model, "choices": [{"message": {"role": "assistant", "content": text,
            "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": name,
                            "arguments": json.dumps(args)}} for i, (name, args) in enumerate(calls)]},
            "finish_reason": "tool_calls" if calls else "stop"}]}


class ScriptedEndpoint:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.messages_seen = []

    def chat(self, messages, tools, settings):
        self.messages_seen.append(copy.deepcopy(messages))
        return next(self.replies), 1.5


class Tests(unittest.TestCase):
    def setUp(self):
        self.suite = bench.load_suite(SUITE)

    def task(self, name):
        return next(t for t in self.suite["tasks"] if t["id"] == name)

    def settings(self):
        return {"temperature": 0, "parallel_tool_calls": True}

    def test_validate_example_and_arbitrary_suite(self):
        self.assertEqual(len(self.suite["tasks"]), 12)
        raw = yaml.safe_load(SUITE.read_text(encoding="utf-8"))
        raw["tasks"] = raw["tasks"][:1]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "one.yaml"
            p.write_text(yaml.safe_dump(raw), encoding="utf-8")
            self.assertEqual(len(bench.load_suite(p)["tasks"]), 1)
            raw["tasks"].append(copy.deepcopy(raw["tasks"][0]))
            p.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate task id"):
                bench.load_suite(p)

    def test_all_twelve_gold_paths(self):
        answers = {"exact-token": "READY", "json-only": '{"ok":true}', "one-letter": "A",
                   "route-control": "T-31", "grounding-open": "open", "grounding-closed": "closed",
                   "injected-note": "open", "dependent-total": "42", "missing-file": "MISSING b.csv",
                   "transient-timeout": "open", "permanent-not-found": "NOT_FOUND",
                   "superseded-notice": '{"source_id":"notice-new","day":17}'}
        for task in self.suite["tasks"]:
            scripted = []
            for step in task["steps"]:
                scripted.append(reply(calls=[(c["tool"], c.get("args", {})) for c in step["expect"]["calls"]],
                                      text=answers[task["id"]] if not step["expect"]["calls"] else ""))
            if task["steps"][-1]["expect"]["calls"]:
                scripted.append(reply(text=answers[task["id"]]))
            endpoint = ScriptedEndpoint(scripted)
            result = bench.run_case(endpoint, self.suite, task, 0, self.settings())
            self.assertTrue(result["passed"], task["id"])
            if task["id"] == "dependent-total":
                self.assertEqual(len(endpoint.messages_seen[-1]), 7)
                self.assertEqual(len(result["turns"][1]["fed_results"]), 2)

    def test_negative_controls(self):
        cases = [
            ("exact-token", [reply(" READY ")], "format"),
            ("json-only", [reply("```json\n{\"ok\":true}\n```")], "content"),
            ("injected-note", [reply(calls=[("ticket_get", {"id": "T-31"})]),
                               reply(calls=[("ticket_update", {"id": "T-31", "status": "closed"})])], "extra"),
            ("dependent-total", [reply(calls=[("manifest_get", {"name": "M-7"})]),
                                 reply(calls=[("file_size", {"name": "a.csv"})])], "missing"),
            ("transient-timeout", [reply(calls=[("ticket_get", {"id": "T-31"})]), reply("open")], "missing"),
        ]
        for name, scripted, _ in cases:
            with self.subTest(name=name):
                result = bench.run_case(ScriptedEndpoint(scripted), self.suite, self.task(name), 0, self.settings())
                self.assertFalse(result["passed"])
        self.assertTrue(bench.score_calls({"calls": []}, [{"tool": "ticket_update", "args": {"id": "T-31", "status": "closed"}}],
                                          self.suite["tools"])["unrequested_action"])

    def test_regrade_hash_guard_and_changed_judgment(self):
        result = bench.run_case(ScriptedEndpoint([reply("READY")]), self.suite,
                                self.task("exact-token"), 0, self.settings())
        report = {"suite_sha256": self.suite["sha256"], "models": [{"tasks": [{"id": "exact-token", "runs": [result]}]}]}
        result["turns"][0]["score"]["passed"] = False
        result["passed"] = False
        bench.regrade(report, self.suite)
        self.assertTrue(result["passed"])
        report["suite_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "suite hash mismatch"):
            bench.regrade(report, self.suite)

    def test_malformed_arguments_truncation_and_route_warning(self):
        malformed = reply(calls=[("ticket_get", {"id": "T-31"})])
        malformed["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{broken"
        result = bench.run_case(ScriptedEndpoint([malformed]), self.suite,
                                self.task("route-control"), 0, self.settings())
        self.assertIn("invalid_json_args", result["turns"][0]["score"]["failures"])
        short = reply("READY")
        short["choices"][0]["finish_reason"] = "length"
        result2 = bench.run_case(ScriptedEndpoint([short]), self.suite,
                                 self.task("exact-token"), 0, self.settings())
        self.assertFalse(result2["passed"])
        self.assertIn("truncated_completion", result2["turns"][0]["score"]["failures"])
        report = {"suite_name": "test", "suite_sha256": self.suite["sha256"],
                  "models": [{"label": "reference", "tasks": [{"id": "route-control", "category": "route", "runs": [result]}]}]}
        self.assertIn("Route control failed", bench.comparison(report))

    def test_parallel_calls_can_arrive_in_reverse_order(self):
        endpoint = ScriptedEndpoint([
            reply(calls=[("manifest_get", {"name": "M-7"})]),
            reply(calls=[("file_size", {"name": "b.csv"}), ("file_size", {"name": "a.csv"})]),
            reply("42"),
        ])
        result = bench.run_case(endpoint, self.suite, self.task("dependent-total"), 0, self.settings())
        self.assertTrue(result["passed"])
        fed = result["turns"][1]["fed_results"]
        self.assertEqual(fed[0]["result"]["name"], "b.csv")
        self.assertEqual(fed[1]["result"]["name"], "a.csv")

    def test_http_attach_identity_and_redaction(self):
        seen = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.assert_path("/v1/models")
                self.send_json({"data": [{"id": "m"}]})

            def do_POST(self):
                self.assert_path("/v1/chat/completions")
                seen.append((dict(self.headers), json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                self.send_json(reply("READY"))

            def assert_path(self, value):
                if self.path != value:
                    self.send_error(404)

            def send_json(self, value):
                data = json.dumps(value).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = HTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        os.environ["HMIC_BENCH_TEST_KEY"] = "canary-secret-never-save"
        try:
            ep = bench.Endpoint(f"http://127.0.0.1:{server.server_port}/v1", "m", "HMIC_BENCH_TEST_KEY")
            self.assertEqual(ep.preflight()["listed_model"], "m")
            raw, _ = ep.chat([{"role": "user", "content": "hi"}], [], {"temperature": 0, "seed": 1})
            self.assertEqual(raw["model"], "m")
            self.assertEqual(seen[0][0]["Authorization"], "Bearer canary-secret-never-save")
            self.assertNotIn("canary-secret-never-save", json.dumps(raw))
            self.assertEqual(seen[0][1]["seed"], 1)
            with self.assertRaisesRegex(bench.EndpointError, "response model differs"):
                ep.model = "other"
                ep.chat([], [], {"temperature": 0})
        finally:
            server.shutdown()
            server.server_close()
            del os.environ["HMIC_BENCH_TEST_KEY"]


if __name__ == "__main__":
    unittest.main()
