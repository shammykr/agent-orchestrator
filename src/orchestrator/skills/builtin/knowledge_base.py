"""Search the support knowledge base for articles matching a problem description."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from orchestrator.contracts import Skill, SkillContext

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the",
    "a",
    "an",
    "is",
    "to",
    "and",
    "or",
    "of",
    "in",
    "on",
    "for",
    "my",
    "i",
    "it",
    "we",
    "our",
    "with",
    "not",
    "be",
    "are",
    "was",
    "this",
    "that",
    "when",
    "at",
    "can",
    "cant",
    "can't",
    "have",
}


class KbSearchInput(BaseModel):
    query: str = Field(min_length=1, description="Free-text problem description")
    top_k: int = Field(default=3, ge=1, le=10)


class KbArticle(BaseModel):
    id: str
    title: str
    category: str
    component: str
    score: float = Field(ge=0, le=1)
    symptoms: list[str]
    steps: list[str]


class KbSearchOutput(BaseModel):
    articles: list[KbArticle]


@lru_cache(maxsize=4)
def _load(path: Path) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = json.loads(path.read_text())
    return data


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall(text.lower()) if t not in _STOP and len(t) > 1}


class KnowledgeBaseSearch(Skill[KbSearchInput, KbSearchOutput]):
    name = "knowledge_base_search"
    description = "Rank knowledge-base articles by keyword overlap with the problem description."
    input_model = KbSearchInput
    output_model = KbSearchOutput

    async def run(self, args: KbSearchInput, ctx: SkillContext) -> KbSearchOutput:
        query = _tokens(args.query)
        scored = []
        for art in _load(ctx.settings.knowledge_base_file):
            vocab = _tokens(" ".join([art["title"], *art["symptoms"], *art["keywords"]]))
            keywords = _tokens(" ".join(art["keywords"]))
            if not query:
                break
            overlap = query & vocab
            # Keyword hits count double: they are the article's curated trigger terms.
            raw = len(overlap) + len(query & keywords)
            if raw:
                scored.append((raw / (len(query) + 2), art))
        scored.sort(key=lambda s: s[0], reverse=True)
        return KbSearchOutput(
            articles=[
                KbArticle(
                    id=a["id"],
                    title=a["title"],
                    category=a["category"],
                    component=a["component"],
                    score=round(min(1.0, s), 3),
                    symptoms=a["symptoms"],
                    steps=a["steps"],
                )
                for s, a in scored[: args.top_k]
            ]
        )
