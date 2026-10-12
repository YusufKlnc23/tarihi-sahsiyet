from __future__ import annotations

import json
from typing import Any, Protocol

from .gemini_client import generate_text
from .schemas import BookMetadataGuess, BookReport, ChunkAnalysis
from .text_cleaning import normalize_for_prompt


class AnalyzerClient(Protocol):
    def guess_metadata(self, filename: str, pdf_metadata: dict[str, str], first_pages: str) -> BookMetadataGuess:
        ...

    def analyze_chunk(self, title: str, author: str, start_page: int, end_page: int, text: str) -> ChunkAnalysis:
        ...

    def synthesize_book(self, title: str, author: str, chunk_analyses: list[ChunkAnalysis]) -> BookReport:
        ...


class GeminiAnalyzer:
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
    def _json_completion(self, system: str, user: str) -> dict:
        content = generate_text(
            api_key=self.api_key,
            model=self.model,
            system=system,
            user=user,
            temperature=0.3,
            max_output_tokens=3500,
        )
        if content.startswith("```json"):
            content = content.removeprefix("```json").strip()
        if content.endswith("```"):
            content = content.removesuffix("```").strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Gemini returned invalid JSON response: {exc}; content={content!r}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("Gemini response did not return a JSON object.")
        return parsed

    def guess_metadata(self, filename: str, pdf_metadata: dict[str, str], first_pages: str) -> BookMetadataGuess:
        system = (
            "Tarih kitabi PDF metadata eslestirme asistanisin. "
            "Sadece gecerli JSON dondur: title, author, year, confidence, evidence."
        )
        user = json.dumps(
            {
                "filename": filename,
                "pdf_metadata": pdf_metadata,
                "first_pages": normalize_for_prompt(first_pages, 7000),
            },
            ensure_ascii=False,
        )
        return BookMetadataGuess.model_validate(self._json_completion(system, user))

    def analyze_chunk(self, title: str, author: str, start_page: int, end_page: int, text: str) -> ChunkAnalysis:
        system = (
            "Tarih metni analiz eden arastirma asistanisin. "
            "Metnin ana argumanlarini, kisi/olay/donem/kavramlari ve tartisma konularini cikar. "
            "Sadece JSON dondur: summary, arguments, people, events, periods, concepts, topics. "
            "topics ogeleri name, importance, pages, rationale alanlarini icersin."
        )
        user = json.dumps(
            {
                "book": {"title": title, "author": author},
                "page_range": [start_page, end_page],
                "text": normalize_for_prompt(text, 18000),
            },
            ensure_ascii=False,
        )
        return ChunkAnalysis.model_validate(self._json_completion(system, user))

    def synthesize_book(self, title: str, author: str, chunk_analyses: list[ChunkAnalysis]) -> BookReport:
        system = (
            "Tarih kitabi analizlerini kitap seviyesinde birlestiren arastirma asistanisin. "
            "Konu agirliklarini yuzde olarak hesapla ve temsilci sayfalari koru. "
            "Sadece JSON dondur: detailed_summary, main_theses, debate_map, evidence, topics. "
            "topics ogeleri name, weight, rationale, representative_pages alanlarini icersin."
        )
        payload = {
            "book": {"title": title, "author": author},
            "chunk_analyses": [analysis.model_dump() for analysis in chunk_analyses],
        }
        return BookReport.model_validate(
            self._json_completion(system, json.dumps(payload, ensure_ascii=False))
        )


class MockAnalyzer:
    """Deterministic local analyzer for tests and dry runs."""

    KEYWORDS = [
        "modernlesme",
        "devlet",
        "milliyetcilik",
        "imparatorluk",
        "osmanli",
        "cumhuriyet",
        "ekonomi",
        "savas",
        "toplum",
        "din",
        "reform",
        "ittihatçilik",
        "İhtilal"
    ]

    def guess_metadata(self, filename: str, pdf_metadata: dict[str, str], first_pages: str) -> BookMetadataGuess:
        title = pdf_metadata.get("title") or filename.rsplit(".", 1)[0]
        author = pdf_metadata.get("author") or "Bilinmeyen yazar"
        return BookMetadataGuess(
            title=title,
            author=author,
            year=None,
            confidence=0.5,
            evidence=["Mock metadata tahmini"],
        )

    def analyze_chunk(self, title: str, author: str, start_page: int, end_page: int, text: str) -> ChunkAnalysis:
        lowered = text.lower()
        topics = []
        for keyword in self.KEYWORDS:
            count = lowered.count(keyword)
            if count:
                topics.append(
                    {
                        "name": keyword.title(),
                        "importance": min(1.0, 0.2 + count / 10),
                        "pages": list(range(start_page, end_page + 1)),
                        "rationale": f"Metinde {count} kez geciyor.",
                    }
                )
        if not topics:
            topics = [
                {
                    "name": "Genel tarihsel tartisma",
                    "importance": 0.4,
                    "pages": list(range(start_page, end_page + 1)),
                    "rationale": "Mock analiz varsayilan konusu.",
                }
            ]
        return ChunkAnalysis(
            summary=normalize_for_prompt(text, 500) or "Bos chunk",
            arguments=["Metnin ana tartismalari chunk seviyesinde ozetlendi."],
            people=[],
            events=[],
            periods=[],
            concepts=[topic["name"] for topic in topics],
            topics=topics,
        )

    def synthesize_book(self, title: str, author: str, chunk_analyses: list[ChunkAnalysis]) -> BookReport:
        totals: dict[str, dict[str, object]] = {}
        for analysis in chunk_analyses:
            for topic in analysis.topics:
                entry = totals.setdefault(
                    topic.name,
                    {"score": 0.0, "pages": set(), "rationale": topic.rationale},
                )
                entry["score"] = float(entry["score"]) + topic.importance
                entry["pages"].update(topic.pages)  # type: ignore[union-attr]
        total_score = sum(float(entry["score"]) for entry in totals.values()) or 1.0
        topics = []
        for name, entry in sorted(totals.items(), key=lambda item: float(item[1]["score"]), reverse=True):
            topics.append(
                {
                    "name": name,
                    "weight": round(float(entry["score"]) / total_score * 100, 2),
                    "rationale": str(entry["rationale"]),
                    "representative_pages": sorted(entry["pages"])[:8],  # type: ignore[arg-type]
                }
            )
        return BookReport(
            detailed_summary="\n\n".join(analysis.summary for analysis in chunk_analyses[:5]),
            main_theses=["Kitapta one cikan tartisma konulari chunk analizlerinden sentezlendi."],
            debate_map=[topic["name"] for topic in topics],
            evidence=["Mock analiz; akademik kullanim icin LLM analiziyle tekrar calistirin."],
            topics=topics,
        )
