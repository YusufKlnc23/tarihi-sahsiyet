from __future__ import annotations

import re

from .config import Settings
from .db import Database
from .retrieval import FigureRetriever, alias_hits
from .schemas import ChatAnswer, Citation, RetrievedChunk
from .text_cleaning import normalize_for_prompt, repair_mojibake


_SOURCE_LINE_RE = re.compile(
    r"^\s*(?:kaynaklar|kaynak|kitap|yazar|sayfa|chunk|parca\s+\d+)\b.*$",
    re.IGNORECASE,
)
_META_PREFIX_RE = re.compile(
    r"^\s*(?:verilen\s+)?(?:kaynaklara|kaynaklarda|metinlere|metinlerde|chunklarda)\s+gore[:,]?\s*",
    re.IGNORECASE,
)


def citation_from_chunk(chunk: RetrievedChunk) -> Citation:
    pages = (
        str(chunk.start_page)
        if chunk.start_page == chunk.end_page
        else f"{chunk.start_page}-{chunk.end_page}"
    )
    return Citation(
        book_title=chunk.book_title,
        author=chunk.author,
        chunk_id=chunk.chunk_id,
        chunk_index=chunk.chunk_index,
        pages=pages,
    )


def format_sources(chunks: list[RetrievedChunk], max_chars: int = 9000) -> str:
    blocks: list[str] = []
    used = 0
    for index, chunk in enumerate(chunks, start=1):
        header = f"[parca {index}]"
        body = normalize_for_prompt(chunk.text, max_chars=1800)
        block = f"{header}\n{body}"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n---\n\n".join(blocks)


def format_judged_topics(topics: list[dict], max_items: int = 8) -> str:
    if not topics:
        return ""
    lines = []
    for index, topic in enumerate(topics[:max_items], start=1):
        lines.append(
            (
                f"{index}. {topic.get('topic_title', '')}\n"
                f"Iddia: {topic.get('claim', '')}\n"
                f"Kanit: {topic.get('evidence', '')}"
            )
        )
    return "\n\n".join(lines)


def build_local_source_answer(
    scope_name: str,
    chunks: list[RetrievedChunk],
    api_unavailable: bool = False,
) -> str:
    lines = [f"{scope_name} hakkinda veri tabanindaki metinlerden cikan ozet:"]
    for index, chunk in enumerate(chunks[:3], start=1):
        preview = normalize_for_prompt(chunk.text, max_chars=850)
        lines.append(f"\n{index}. {preview}")
    if api_unavailable:
        lines.append(
            "\nNot: Gemini API su an kullanilamadigi icin cevap, bulunan metin "
            "parcalarindan yerel olarak hazirlandi."
        )
    return "\n".join(lines)


def clean_generated_answer(answer: str) -> str:
    answer = repair_mojibake(answer).strip()
    answer = _META_PREFIX_RE.sub("", answer)
    lines = [
        line.rstrip()
        for line in answer.splitlines()
        if not _SOURCE_LINE_RE.match(line)
    ]
    return "\n".join(lines).strip()



class FigureChatService:
    def __init__(self, db: Database, settings: Settings) -> None:
        self.db = db
        self.settings = settings
        self.retriever = FigureRetriever(db)

    def infer_figure_from_question(self, question: str) -> dict | None:
        if self.db is None:
            return None
        candidates: list[tuple[int, dict]] = []
        for figure in self.db.list_figures():
            aliases = [figure["name"], *(figure.get("aliases") or [])]
            hits = alias_hits(question, aliases)
            if hits:
                candidates.append((hits, figure))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
            return None
        return candidates[0][1]

    def answer(
        self,
        question: str,
        figure_id: int | None = None,
        use_llm: bool = True,
    ) -> ChatAnswer:
        figure = self.db.get_figure(figure_id) if figure_id is not None else None
        if figure_id is None:
            figure = self.infer_figure_from_question(question)
            if figure:
                figure_id = int(figure["id"])
        if figure_id is not None and not figure:
            return ChatAnswer(answer="Secilen sahsiyet veritabaninda bulunamadi.")

        if figure:
            chunks = self.retriever.retrieve(figure_id=figure_id, question=question)
            scope_name = figure["name"]
            scope_period = figure.get("period") or "Bilinmiyor"
        else:
            chunks = self.retriever.retrieve_general(question=question)
            scope_name = "Tum kaynaklar"
            scope_period = "Genel arama"

        if not chunks:
            return ChatAnswer(
                answer=(
                    "Bu veri icinde soruyu cevaplayacak kaynak chunk bulunamadi. "
                    "Once `python -m tarih_pdf_analyzer ingest data/pdfs --force` "
                    "komutuyla PDF/TXT kaynaklarini yukleyin."
                )
            )

        citations = [citation_from_chunk(chunk) for chunk in chunks]
        judged_topics = []
        if hasattr(self.db, "get_approved_topic_context"):
            try:
                judged_topics = self.db.get_approved_topic_context(
                    [chunk.chunk_id for chunk in chunks]
                )
            except Exception:
                judged_topics = []
        api_key = self.settings.gemini_api_key
        if not use_llm or not api_key:
            return ChatAnswer(
                answer=build_local_source_answer(scope_name, chunks),
                citations=citations,
            )

        system = (
            "Turk tarihi kaynaklari uzerinden cevap veren bir arastirma asistanisin. "
            "Sadece verilen kaynak chunk'lara dayan. Kaynaklarda yoksa bunu acikca soyle. "
            "Secili sahsiyet kaynaklarda acikca gecmiyorsa tahmin yurutme. "
            "Kisi adlari benzerse veya kanit zayifsa belirsiz oldugunu soyle. "
            "Cevapta kitap adi, yazar adi, sayfa numarasi, chunk id veya kaynak listesi yazma. "
            "`Verilen kaynaklara gore`, `metinlerde`, `kaynaklarda` gibi meta girisler kullanma. "
            "Onayli tartisma konulari varsa cevabi bu konularla hizala. "
            "Soru belirli bir konuya odaklaniyorsa genel biyografi yazma; dogrudan o konuyu cevapla. "
            "Once sorunun net cevabini ver, sonra gerekli baglami ve gerekceyi ekle. "
            "Turkce karakterleri dogal ve dogru kullan. "
            "Kullaniciya dogrudan, akici ve anlamli cevabi ver. "
            "Kisa gecme; 4-6 paragraf yaz, gerekiyorsa maddeli aciklama kullan."
        )
        user = (
            f"Kapsam: {scope_name}\n"
            f"Donem: {scope_period}\n"
            f"Soru: {question}\n\n"
            f"Onayli tartisma konulari:\n{format_judged_topics(judged_topics) or '-'}\n\n"
            f"Arka plan parcalari:\n{format_sources(chunks)}"
        )

        try:
            from .gemini_client import generate_text

            answer = generate_text(
                api_key=api_key,
                model=self.settings.gemini_model,
                system=system,
                user=user,
                temperature=0.2,
                max_output_tokens=3200,
            )
        except Exception:
            answer = build_local_source_answer(
                scope_name,
                chunks,
                api_unavailable=True,
            )
        return ChatAnswer(answer=clean_generated_answer(answer), citations=citations)
