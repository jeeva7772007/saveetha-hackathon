from http.server import BaseHTTPRequestHandler
import json
import sys
import os

# Make api/ importable
sys.path.insert(0, os.path.dirname(__file__))
import model_utils


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            data   = json.loads(body) if body else {}
        except Exception:
            self._respond(400, {"error": "Invalid JSON body."})
            return

        prompt = str(data.get("prompt", "")).strip()
        if not prompt:
            self._respond(400, {"error": "Missing 'prompt' in request body."})
            return

        try:
            result = model_utils.predict(prompt)
            if "error" in result:
                self._respond(400, result)
            else:
                self._respond(200, result)
        except FileNotFoundError as e:
            self._respond(503, {
                "error": "Model not found. Please train the model first.",
                "details": str(e)
            })
        except Exception as e:
            self._respond(500, {
                "error": "An internal error occurred during analysis.",
                "details": str(e)
            })

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _respond(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self._cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
