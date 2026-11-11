"""Mock tool: search_knowledge_base — keyword retrieval over internal docs."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field

from ops_harness.tools.base import Tool


class KBArgs(BaseModel):
    query: str = Field(..., min_length=1, description="Free-text search query")
    top_k: int = Field(3, ge=1, le=10, description="Maximum number of hits to return")


class KBHit(BaseModel):
    id: str
    title: str
    snippet: str
    score: float


class KBOutput(BaseModel):
    results: list[KBHit]


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _load_articles(data_dir: Path) -> list[dict]:
    with open(data_dir / "kb_articles.json", encoding="utf-8") as f:
        return json.load(f)


def make_kb_tool(data_dir: Path, timeout_s: float = 10.0, max_retries: int = 2,
                 backoff_base_s: float = 0.25) -> Tool:
    def fn(args: KBArgs) -> dict:
        articles = _load_articles(data_dir)
        q_tokens = _tokenize(args.query)
        hits = []
        for art in articles:
            doc_tokens = _tokenize(
                f"{art.get('title', '')} {' '.join(art.get('tags', []))} {art.get('body', '')}"
            )
            overlap = q_tokens & doc_tokens
            if not overlap:
                continue
            score = round(len(overlap) / max(len(q_tokens), 1), 4)
            hits.append(
                KBHit(
                    id=art["id"],
                    title=art["title"],
                    snippet=art["body"][:220],
                    score=score,
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return {"results": [h.model_dump() for h in hits[: args.top_k]]}

    return Tool(
        name="search_knowledge_base",
        description="Search internal documentation (runbooks, on-call guides). "
        "Returns the most relevant article snippets.",
        args_model=KBArgs,
        fn=fn,
        output_model=KBOutput,
        timeout_s=timeout_s,
        max_retries=max_retries,
        backoff_base_s=backoff_base_s,
    )
