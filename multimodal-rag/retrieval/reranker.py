from __future__ import annotations

from typing import Any, Dict, List


class Reranker:
    """Optional reranking step to prioritize the most relevant documents."""

    def rerank(self, results: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        if not results or "documents" not in results:
            return results

        documents = results["documents"][0]
        distances = results.get("distances", [[0.0 for _ in documents]])[0]

        combined = sorted(
            zip(documents, distances),
            key=lambda item: item[1],
        )[:top_k]

        return {
            "documents": [list(doc for doc, _ in combined)],
            "distances": [list(distance for _, distance in combined)],
        }
