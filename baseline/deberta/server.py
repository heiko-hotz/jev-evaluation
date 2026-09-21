"""Loopback-only CPU classification service; no inference during startup."""
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "model"
MANIFEST = json.loads((MODEL / "manifest.json").read_text())
torch.set_num_threads(int(os.environ.get("DEBERTA_THREADS", "4")))
torch.set_num_interop_threads(1)
tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, local_files_only=True, trust_remote_code=False,
    use_safetensors=True, torch_dtype=torch.float32,
)
model.eval()
classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer, device=-1)
if any(p.device.type != "cpu" for p in model.parameters()):
    raise RuntimeError("CPU-only deployment required")


class Handler(BaseHTTPRequestHandler):
    # Single request at a time; no request bodies or predictions in access logs.
    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def log_message(self, *args):
        pass

    def send_json(self, status, value):
        payload = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path != "/health":
            return self.send_json(404, {"error": "not found"})
        self.send_json(200, {"status": "loaded", "device": "cpu",
                             "model": MANIFEST["repo_id"], "revision": MANIFEST["revision"]})

    def do_POST(self):
        if self.path != "/classify":
            return self.send_json(404, {"error": "not found"})
        try:
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("Transfer-Encoding is not supported")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                raise ValueError("Content-Length must be between 1 and 65536")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("JSON object required")
            text = body.get("inputs")
            params = body.get("parameters", {})
            if not isinstance(text, str) or not text.strip() or not isinstance(params, dict):
                raise ValueError("Nonempty inputs string and parameters object required")
            labels = params.get("candidate_labels")
            if (not isinstance(labels, list) or not 1 <= len(labels) <= 151
                    or any(not isinstance(x, str) or not x.strip() or len(x) > 512 for x in labels)
                    or len(set(labels)) != len(labels)):
                raise ValueError("Provide 1 to 151 unique nonempty candidate_labels, at most 512 characters each")
            multi = params.get("multi_label", False)
            template = params.get("hypothesis_template", "This example is {}.")
            if not isinstance(multi, bool):
                raise ValueError("multi_label must be boolean")
            if not isinstance(template, str) or len(template) > 1024 or template.count("{}") != 1:
                raise ValueError("hypothesis_template must contain exactly one {} placeholder")
            try:
                hypotheses = [template.format(label) for label in labels]
            except (ValueError, KeyError, IndexError):
                raise ValueError("Invalid hypothesis_template") from None
            # Reject oversized pairs; do not silently truncate evidence.
            if any(len(tokenizer(text, h, truncation=False)["input_ids"]) > 512 for h in hypotheses):
                raise ValueError("Text plus each label hypothesis must fit within 512 tokens")
        except (ValueError, TypeError, UnicodeError):
            return self.send_json(400, {"error": "Invalid request; see the documented input limits"})
        try:
            with torch.inference_mode():
                result = classifier(text, candidate_labels=labels, hypothesis_template=template,
                                    multi_label=multi, batch_size=1)
            self.send_json(200, {"labels": result["labels"], "scores": result["scores"],
                                 "model": MANIFEST["repo_id"], "revision": MANIFEST["revision"]})
        except Exception:
            self.send_json(500, {"error": "Classification failed"})


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 8091), Handler)
    print("DeBERTa loaded on CPU; listening on 127.0.0.1:8091; no startup inference", flush=True)
    server.serve_forever()
