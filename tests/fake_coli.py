"""Stand-in for `coli serve`: OpenAI-style gateway that echoes the last user message."""

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def doctor_or_convert(argv) -> bool:
    """`doctor`: model is ready iff <dir>/ready exists. `convert`: writes <outdir>/ready."""
    if argv[:1] == ["doctor"]:
        model = argv[argv.index("--model") + 1]
        checks = [{"id": "engine.binary", "status": "fail", "summary": "engine is not built"}]
        if not os.path.exists(os.path.join(model, "ready")) and not os.environ.get("FAKE_REQUIRE_READY"):
            checks.append({"id": "model.shards", "status": "fail", "summary": "not in Colibri format"})
        print(json.dumps({"status": "error", "checks": checks}))
        sys.exit(1)
    if argv[:1] == ["convert"]:
        out = argv[argv.index("--model") + 1]
        if os.path.exists(out):
            sys.exit("output dir exists")
        os.makedirs(out)
        open(os.path.join(out, "ready"), "w").write(argv[argv.index("--repo") + 1])
        sys.exit(int(os.environ.get("FAKE_CONVERT_RC", "0")))
    return False


def main() -> None:
    doctor_or_convert(sys.argv[1:])
    args = sys.argv[2:] if sys.argv[1:2] == ["serve"] else sys.argv[1:]
    opts = dict(zip(args[::2], args[1::2]))
    model_id, port = opts["--model-id"], int(opts["--port"])
    ready = os.path.exists(os.path.join(opts["--model"], "ready"))
    delay = float(opts.get("--delay", "0"))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            raw = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            self._send(200, {"status": "ok"}) if self.path == "/health" else self._send(404, {})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            chat = self.path == "/v1/chat/completions"
            prompt = body["messages"][-1]["content"] if chat else body["prompt"]
            if os.environ.get("FAKE_REQUIRE_READY") and not ready:
                return self._send(500, {"error": {"message": "engine crashed"}})
            if prompt == "boom":
                return self._send(500, {"error": {"message": "engine exploded"}})
            time.sleep(delay)
            words = [f"{model_id}:", *prompt.split()]
            if not body.get("stream"):
                text = " ".join(words)
                choice = {"message": {"role": "assistant", "content": text}} if chat else {"text": text}
                return self._send(200, {"choices": [{**choice, "finish_reason": "stop"}],
                                        "usage": {"prompt_tokens": 3, "completion_tokens": len(words)}})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for w in words:
                piece = w + " "
                choice = {"delta": {"content": piece}} if chat else {"text": piece}
                self.wfile.write(f"data: {json.dumps({'choices': [choice]})}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


main()
