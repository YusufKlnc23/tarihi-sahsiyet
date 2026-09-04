from __future__ import annotations

import re

from .models import PageText, TextChunk


def estimate_tokens(text: str) -> int:
    words = re.findall(r"\S+", text)
    return max(1, int(len(words) * 1.35))


def split_oversized_text(text: str, max_tokens: int) -> list[str]:
    words = text.split()
    if estimate_tokens(text) <= max_tokens:
        return [text]

    max_words = max(100, int(max_tokens / 1.35))
    return [" ".join(words[index : index + max_words]) for index in range(0, len(words), max_words)]


def _overlap_tail(text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0:
        return ""
    words = text.split()
    overlap_words = max(1, int(overlap_tokens / 1.35))
    return " ".join(words[-overlap_words:])


def build_discussion_question_chunks(source_chunks: list[TextChunk]) -> list[TextChunk]:
    questions: list[TextChunk] = []
    for index, chunk in enumerate(source_chunks):
        if not chunk.text.strip():
            continue
        topic_terms = [
            term
            for term in re.findall(r"\b[\wğüşöçıİĞÜŞÖÇ]{4,}\b", chunk.text, flags=re.UNICODE)
            if term.lower() not in {"bu", "metin", "olarak", "için", "ile", "ve", "bir", "vardir"}
        ]
        topic = " ".join(topic_terms[:4]) or "bu konu"
        question_text = (
            f"Tartişma sorusu: {topic} konusu bu kaynakta nasıl ele alınıyor ve hangi sonuçlar ortaya çıkıyor?"
        )
        questions.append(
            TextChunk(
                chunk_index=chunk.chunk_index + len(source_chunks),
                start_page=chunk.start_page,
                end_page=chunk.end_page,
                text=question_text,
                token_count=estimate_tokens(question_text),
                page_numbers=list(chunk.page_numbers or [chunk.start_page]),
                chunk_type="discussion_question",
            )
        )
    return questions


def chunk_pages(
    pages: list[PageText],
    max_tokens: int = 1500,
    overlap_tokens: int = 200,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    current_parts: list[str] = []
    current_pages: list[int] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current_parts, current_pages, current_tokens
        if not current_parts:
            return
        text = "\n\n".join(part for part in current_parts if part.strip()).strip()
        if not text:
            current_parts, current_pages, current_tokens = [], [], 0
            return
        page_numbers = sorted(set(current_pages))
        chunks.append(
            TextChunk(
                chunk_index=len(chunks) + 1,
                start_page=page_numbers[0],
                end_page=page_numbers[-1],
                text=text,
                token_count=estimate_tokens(text),
                page_numbers=page_numbers,
            )
        )
        overlap = _overlap_tail(text, overlap_tokens)
        if overlap:
            current_parts = [overlap]
            current_pages = [page_numbers[-1]]
            current_tokens = estimate_tokens(overlap)
        else:
            current_parts, current_pages, current_tokens = [], [], 0

    for page in pages:
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", page.text) if p.strip()]
        if not paragraphs and page.text.strip():
            paragraphs = [page.text.strip()]
        for paragraph in paragraphs:
            for part in split_oversized_text(paragraph, max_tokens):
                part_tokens = estimate_tokens(part)
                if current_parts and current_tokens + part_tokens > max_tokens:
                    flush()
                current_parts.append(part)
                current_pages.append(page.page_number)
                current_tokens += part_tokens
    flush()

    return chunks
