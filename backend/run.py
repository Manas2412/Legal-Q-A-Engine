"""
Entry point for the Legal Q&A Engine backend.

Usage:
    python run.py                   # start with defaults
    python run.py --port 8080       # custom port
    python run.py --reload          # hot reload for development

The React frontend is a completely separate project.
This backend exposes a REST + SSE API at http://localhost:8000.
Set CORS_ORIGINS in .env to your frontend URL.
"""
from __future__ import annotations
import argparse
import uvicorn
from config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Legal Q&A Engine — Backend API")
    parser.add_argument("--host",   default=settings.host,  help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port",   default=settings.port,  type=int, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true",    help="Enable hot reload (development only)")
    parser.add_argument("--workers",default=1,              type=int, help="Number of worker processes")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Legal Q&A Engine — Indian Law")
    print(f"  API:  http://{args.host}:{args.port}")
    print(f"  Docs: http://{args.host}:{args.port}/docs")
    print(f"  CORS: {settings.cors_allow_origins}")
    print(f"  LLM:  {settings.llm_model} @ {settings.ollama_base_url}")
    print(f"{'='*60}\n")

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="debug" if settings.debug else "info",
    )


if __name__ == "__main__":
    main()