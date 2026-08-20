from __future__ import annotations

import json
from typing import Protocol

from .gemini_client import generate_text
from .schemas import (
    DebateJudgeResponse,
    DebateTopicCandidate,
    DebateTopicJudgement,
    JudgedDebateTopic,
)
from .text_cleaning import normalize_for_prompt


class TopicJudgeClient(Protocol):
    def evaluate_chunk(self, book: dict, chunk: dict) -> DebateJudgeResponse:
        ...


def _parse_json_object(content: str) -> dict:
    content = content.strip()
    if content.startswith("```json"):
        content = content.removeprefix("```json").strip()
    if content.endswith("```"):
        content = content.removesuffix("```").strip()
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise RuntimeError("Judge response did not return a JSON object.")
    return parsed


class GeminiTopicJudge:
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def evaluate_chunk(self, book: dict, chunk: dict) -> DebateJudgeResponse:
        system = (
            "Tarih metinlerinde tartisma konusu kalite yargici olarak calis. "
            "Sadece verilen chunk'a dayan. Tarihsel dogruyu dis bilgiden yargilama; "
            "yalnizca aday konunun bu chunk tarafindan desteklenip desteklenmedigini degerlendir. "
            "Genel biyografi basliklarini ele; tartisma degeri olan kisi, olay, donem, fikir veya yorum ayriliklarini sec. "
            "Sadece gecerli JSON dondur: {\"topics\": [{\"candidate\": {...}, \"judgement\": {...}}]}. "
            "candidate alanlari: topic_title, claim, people, events, periods, evidence, confidence. "
            "judgement alanlari: approved, relevance_score, evidence_score, hallucination_risk, debate_value, action, reason. "
            "action sadece keep, merge, revise, reject veya review olabilir."
        )
        user = json.dumps(
            {
                "book": {
                    "id": book.get("id"),
                    "title": book.get("title"),
                    "author": book.get("author"),
                },
                "chunk": {
                    "id": chunk.get("id"),
                    "chunk_index": chunk.get("chunk_index"),
                    "page_range": [chunk.get("start_page"), chunk.get("end_page")],
                    "text": normalize_for_prompt(str(chunk.get("text") or ""), 16000),
                },
                "rubric": {
                    "approve_when": [
                        "relevance_score >= 75",
                        "evidence_score >= 70",
                        "hallucination_risk <= 30",
                        "debate_value >= 60",
                    ],
                    "reject_when": [
                        "topic is generic biography only",
                        "claim is not directly supported by the chunk",
                        "topic title is too broad",
                    ],
                },
            },
            ensure_ascii=False,
        )
        content = generate_text(
            api_key=self.api_key,
            model=self.model,
            system=system,
            user=user,
            temperature=0.05,
            max_output_tokens=2600,
        )
        return DebateJudgeResponse.model_validate(_parse_json_object(content))


class MockTopicJudge:
    KEYWORDS = ("tartisma", "ihtilal", "darbe", "mesrutiyet", "ittihat", "savas", "reform")

    def evaluate_chunk(self, book: dict, chunk: dict) -> DebateJudgeResponse:
        text = normalize_for_prompt(str(chunk.get("text") or ""), 700)
        lowered = text.casefold()
        keyword = next((item for item in self.KEYWORDS if item in lowered), "genel tarihsel tartisma")
        candidate = DebateTopicCandidate(
            topic_title=keyword.title(),
            claim=f"Chunk icinde {keyword} eksenli bir tartisma aday konusu var.",
            people=[],
            events=[],
            periods=[],
            evidence=text[:280] or "Mock evidence",
            confidence=0.6,
        )
        judgement = DebateTopicJudgement(
            approved=True,
            relevance_score=75,
            evidence_score=70,
            hallucination_risk=0,5,
            debate_value=65,
            action="keep",
            reason="Mock judge varsayilan onayi.",
        )
        return DebateJudgeResponse(topics=[JudgedDebateTopic(candidate=candidate, judgement=judgement)])


def judge_book_topics(
    db,
    client: TopicJudgeClient,
    book: dict,
    model_name: str,
    limit: int | None = None,
) -> int:
    chunks = [
        chunk
        for chunk in db.get_chunks(book["id"])
        if chunk.get("chunk_type", "source") == "source"
    ]
    if limit is not None:
        chunks = chunks[:limit]

    written = 0
    for chunk in chunks:
        response = client.evaluate_chunk(book, chunk)
        db.replace_topic_judgements_for_chunk(chunk["id"], response.topics, model_name)
        written += len(response.topics)
    return written
