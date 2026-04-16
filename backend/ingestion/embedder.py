from __future__ import annotations
import httpx
import time
from typing import Optional
from config import settings

def embed_text(text: str, retries: int = 3) -> Optional[list[float]]:
    """
    Get embedding from nomic-embed-text via Ollama.
    Returns None on failure after retries.
    """
    
    payload = {"model": settings.embed_model, "prompt": text}
    
    for attempt in range(retries):
        try:
            resp = httpx.post(
                f"{settings.ollama_base_url}/api/embeddings",
                json=payload,
                timeout=60.0
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
        except Exception as exec:
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                print(f"[Embedder] Failed after {retries} attempts: {exec}")
                return None
            
def embed_match(texts: list[str], batch_size: int = 8) -> list[Optional[list[float]]]:
    """Enbed a list of text in batches"""
    results: list[Optional[list[float]]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for text in batch:
            results.append(embed_text(text))
        if i + batch_size < len(texts):
            time.sleep(0.1)
            
    return results
