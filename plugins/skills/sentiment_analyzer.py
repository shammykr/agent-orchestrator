"""Example plugin skill, loaded from plugins/skills/ at start-up: no change to the orchestrator.

Scores customer sentiment and frustration so the summary agent can adapt its tone.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from orchestrator.contracts import Skill, SkillContext

_NEGATIVE = re.compile(
    r"\b(angry|frustrat\w*|unacceptable|terrible|awful|ridiculous|still|again|worst|furious|cancel\w*|"
    r"urgent|asap|immediately|nobody|useless|broken|losing|lost)\b",
    re.IGNORECASE,
)
_POSITIVE = re.compile(r"\b(thanks?|thank you|appreciate|great|please|love|helpful)\b", re.IGNORECASE)


class SentimentInput(BaseModel):
    text: str = Field(min_length=1)


class SentimentOutput(BaseModel):
    sentiment: Literal["negative", "neutral", "positive"]
    score: float = Field(ge=-1, le=1)
    frustration_signals: list[str]


class SentimentAnalyzer(Skill[SentimentInput, SentimentOutput]):
    name = "sentiment_analyzer"
    version = "0.9"
    description = "Lexicon-based sentiment and frustration signals for tone adaptation (plugin example)."
    input_model = SentimentInput
    output_model = SentimentOutput

    async def run(self, args: SentimentInput, ctx: SkillContext) -> SentimentOutput:
        negatives = [m.lower() for m in _NEGATIVE.findall(args.text)]
        positives = _POSITIVE.findall(args.text)
        exclaims = min(args.text.count("!"), 3)
        raw = len(positives) - len(negatives) - 0.5 * exclaims
        score = max(-1.0, min(1.0, raw / 4))
        sentiment: Literal["negative", "neutral", "positive"] = (
            "negative" if score <= -0.25 else "positive" if score >= 0.25 else "neutral"
        )
        return SentimentOutput(sentiment=sentiment, score=round(score, 2), frustration_signals=sorted(set(negatives)))
