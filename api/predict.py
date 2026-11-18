"""Vercel Serverless Function: POST /api/predict
Body: {"values": [v1, v2, v3]}  ->  RNN / LSTM / GRU کی اگلی ویلیو
صرف NumPy استعمال ہوتا ہے (TensorFlow Vercel پر بہت بھاری ہے)۔
"""
import json
import os
from http.server import BaseHTTPRequestHandler

import numpy as np

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_weights.json")
with open(_PATH) as f:
    CFG = json.load(f)
W = {name: {k: np.array(v) for k, v in params.items()} for name, params in CFG["models"].items()}
H = CFG["hidden"]


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def run(kind, p, seq):
    h = np.zeros(H)
    c = np.zeros(H)
    for x in seq:
        if kind == "rnn":
            h = np.tanh(x * p["Wx"][0] + h @ p["Wh"] + p["b"])
        elif kind == "lstm":
            z = x * p["Wx"][0] + h @ p["Wh"] + p["b"]
            i, fg, g, o = sig(z[:H]), sig(z[H:2 * H]), np.tanh(z[2 * H:3 * H]), sig(z[3 * H:])
            c = fg * c + i * g
            h = o * np.tanh(c)
        else:  # gru
            zx = x * p["Wx"][0] + p["b"]
            zr = h @ p["Wh"]
            z = sig(zx[:H] + zr[:H])
            r = sig(zx[H:2 * H] + zr[H:])
            n = np.tanh(zx[2 * H:] + (r * h) @ p["Un"])
            h = (1 - z) * h + z * n
    return float((h @ p["Wy"] + p["by"])[0])


def predict(values):
    lo, hi = CFG["min"], CFG["max"]
    seq = [(v - lo) / (hi - lo) for v in values]
    out = {}
    for kind in ("rnn", "lstm", "gru"):
        out[kind] = round(run(kind, W[kind], seq) * (hi - lo) + lo, 2)
    return {"predictions": out, "rmse": CFG["rmse"], "best": CFG["best"]}


class handler(BaseHTTPRequestHandler):
    def _send(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            values = json.loads(self.rfile.read(n))["values"]
            values = [float(v) for v in values]
            if len(values) != CFG["window"]:
                raise ValueError(f"{CFG['window']} values required")
            self._send(200, predict(values))
        except Exception as e:
            self._send(400, {"error": str(e)})
