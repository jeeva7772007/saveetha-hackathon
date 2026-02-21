from http.server import BaseHTTPRequestHandler
import json
import os
import pickle


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
            with open(os.path.join(base, "disease_info.pkl"), "rb") as f:
                disease_info = pickle.load(f)
            diseases = [
                {
                    "name": name,
                    "risk_level": info["risk_level"],
                    "is_emergency": info["is_emergency"]
                }
                for name, info in sorted(disease_info.items())
            ]
            self._respond(200, {"diseases": diseases, "count": len(diseases)})
        except Exception as e:
            self._respond(500, {"error": str(e)})

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _respond(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self._cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
