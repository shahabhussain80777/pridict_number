import json
import os

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

BASE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE, "api", "model_weights.json")) as f:
    CFG = json.load(f)
W = {n: {k: np.array(v) for k, v in p.items()} for n, p in CFG["models"].items()}
H = CFG["hidden"]

app = Flask(__name__)


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


@app.route("/")
def home():
    return send_from_directory(os.path.join(BASE, "public"), "index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        values = [float(v) for v in request.get_json(force=True)["values"]]
        if len(values) != CFG["window"]:
            raise ValueError(f"{CFG['window']} values required")
        lo, hi = CFG["min"], CFG["max"]
        seq = [(v - lo) / (hi - lo) for v in values]
        preds = {k: round(run(k, W[k], seq) * (hi - lo) + lo, 2) for k in ("rnn", "lstm", "gru")}
        return jsonify(predictions=preds, rmse=CFG["rmse"], best=CFG["best"])
    except Exception as e:
        return jsonify(error=str(e)), 400
