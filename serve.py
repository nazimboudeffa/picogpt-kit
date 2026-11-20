"""Petit serveur HTTP local pour interroger le modèle sans Ollama.

Usage :
    python serve.py
    curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d "{\"prompt\":\"What is 2+2?\\n\"}"
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from model_utils import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_NO_REPEAT_NGRAM_SIZE,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    MODEL_DIR,
    generate,
    load_model,
)
from rag import DEFAULT_MIN_SCORE, answer_with_rag


def make_handler(tokenizer, model):
    """Construit un handler HTTP qui garde le modèle chargé en mémoire."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/generate":
                self.send_error(404, "Utilisez POST /generate")
                return

            length = int(self.headers.get("Content-Length", 0))
            try:
                data = json.loads(self.rfile.read(length))
                prompt = data["prompt"]
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("Le champ 'prompt' doit être une chaîne non vide.")
                gen_kwargs = {
                    "max_new_tokens": int(
                        data.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
                    ),
                    "do_sample": bool(data.get("do_sample", True)),
                    "temperature": float(data.get("temperature", DEFAULT_TEMPERATURE)),
                    "top_p": float(data.get("top_p", DEFAULT_TOP_P)),
                    "top_k": int(data.get("top_k", DEFAULT_TOP_K)),
                    "repetition_penalty": float(
                        data.get("repetition_penalty", DEFAULT_REPETITION_PENALTY)
                    ),
                    "no_repeat_ngram_size": int(
                        data.get("no_repeat_ngram_size", DEFAULT_NO_REPEAT_NGRAM_SIZE)
                    ),
                }
                if bool(data.get("use_rag", False)):
                    text = answer_with_rag(
                        tokenizer,
                        model,
                        prompt,
                        k=int(data.get("rag_k", 1)),
                        min_score=float(data.get("min_score", DEFAULT_MIN_SCORE)),
                        **gen_kwargs,
                    )
                else:
                    text = generate(tokenizer, model, prompt, **gen_kwargs)
            except Exception as exc:
                self._send_json(400, {"error": str(exc)})
                return

            self._send_json(200, {"generated_text": text})

        def _send_json(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Sert le modèle sur HTTP en local")
    parser.add_argument("--model-dir", default=MODEL_DIR, help="dossier du modèle entraîné")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_dir)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(tokenizer, model))
    print(f"Serveur prêt : http://{args.host}:{args.port}/generate")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
