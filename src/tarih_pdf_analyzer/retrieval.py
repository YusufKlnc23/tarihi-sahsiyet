from __future__ import annotations

import re
import unicodedata

from .db import Database
from .schemas import RetrievedChunk
from .text_cleaning import repair_mojibake


_WORD_RE = re.compile(r"\w{3,}", re.UNICODE)
_TR_MAP = str.maketrans(
    {
        "ç": "c",
        "Ç": "c",
        "ğ": "g",
        "Ğ": "g",
        "ı": "i",
        "I": "i",
        "İ": "i",
        "ö": "o",
        "Ö": "o",
        "ş": "s",
        "Ş": "s",
        "ü": "u",
        "Ü": "u",
    }
)
_QUESTION_STOP_TERMS = {
    'fakat','lakin','ancak','acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç', 'birşey', 'biz', 'bu', 'çok', 'çünkü', 'da', 'daha', 'de', 'defa', 'diye', 'eğer', 'en', 'gibi', 'hem', 'hep', 'hepsi', 'her', 'hiç', 'için', 'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mu',
    'mü', 'nasıl', 'ne', 'neden', 'nerde', 'nerede', 'nereye', 'niçin', 'niye', 'o', 'sanki', 'şey', 'siz', 'şu', 'tüm', 've', 'veya', 'ya', 'yani'
    "acaba",
    "anlat",
    "anlatiliyor",
    "anlatilir",
    "bilgi",
    "cevap",
    "degerlendiriliyor",
    "detay",
    "detayli",
    "hakkinda",
    "hangi",
    "kaynak",
    "kaynaklarda",
    "kimdir",
    "kisaca",
    "nda",
    "nde",
    "metinlerde",
    "nedir",
    "nasil",
    "neydi",
    "neden",
    "olarak",
    "rolu",
    "sence",
    "soyle",
    "ver",
}
_ALIAS_TOKEN_STOP_TERMS = {"bey", "han", "pasa", "paşa", "sultan"}
_SEARCH_CHAR_VARIANTS = {
    "c": ("ç",),
    "g": ("ğ",),
    "i": ("ı",),
    "o": ("ö",),
    "s": ("ş",),
    "u": ("ü",),
}
_FIGURE_SOURCE_PRIORITIES = {
    "enver-pasa": (
        ("turkiyenin yakin tarihi", 89, 500),
        ("turkiye'nin yakin tarihi", 89, 500),
    ),
    "mehmed-resad": (
        ("ittihatterrakikiskacindabirsultan", None, 200),
        ("ittihat terraki kiskacinda bir sult", None, 80),
        ("ittihat terakki kiskacinda bir sultan", None, 80),
    )
}


def normalize_match(value: str) -> str:
    value = repair_mojibake(value)
    value = value.translate(_TR_MAP)
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_value.casefold()


def question_terms(question: str, max_terms: int = 8) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    normalized_question = normalize_match(question)
    for match in _WORD_RE.finditer(normalized_question):
        term = match.group(0)
        if term in _QUESTION_STOP_TERMS or term in seen:
            continue
        seen.add(term)
        terms.append(term)
        if len(terms) >= max_terms:
            break
    return terms


def terminal_dt_variants(value: str, max_variants: int = 16) -> list[str]:
    words = value.split()
    if not words:
        return []

    variants = [""]
    for word in words:
        options = [word]
        if len(word) > 3 and word.endswith("d"):
            options.append(f"{word[:-1]}t")
        elif len(word) > 3 and word.endswith("t"):
            options.append(f"{word[:-1]}d")

        next_variants: list[str] = []
        for prefix in variants:
            for option in options:
                next_variants.append(f"{prefix} {option}".strip())
                if len(next_variants) >= max_variants:
                    break
            if len(next_variants) >= max_variants:
                break
        variants = next_variants

    return variants


def search_variants(term: str, max_variants: int = 16) -> list[str]:
    variants = {term}
    for spelling in terminal_dt_variants(term, max_variants=max_variants):
        variants.add(spelling)
    for source, replacements in _SEARCH_CHAR_VARIANTS.items():
        if all(source not in variant for variant in variants):
            continue
        current = list(variants)
        for value in current:
            for replacement in replacements:
                variants.add(value.replace(source, replacement))
                if len(variants) >= max_variants:
                    return sorted(variants)
    return sorted(variants)


def search_patterns(aliases: list[str], terms: list[str]) -> list[str]:
    patterns: list[str] = []

    def add(value: str) -> None:
        value = value.strip()
        if len(value) >= 3 and value not in patterns:
            patterns.append(value)

    for alias in aliases:
        add(alias)
        normalized_alias = normalize_match(alias).strip()
        add(normalized_alias)
        for variant in search_variants(normalized_alias):
            add(variant)
        for match in _WORD_RE.finditer(normalized_alias):
            if match.group(0) in _ALIAS_TOKEN_STOP_TERMS:
                continue
            for variant in search_variants(match.group(0)):
                add(variant)

    for term in terms:
        for variant in search_variants(term):
            add(variant)

    return patterns[:80]


def normalized_aliases(aliases: list[str]) -> list[str]:
    cleaned: list[str] = []
    for alias in aliases:
        normalized = normalize_match(alias).strip()
        if not normalized or len(normalized) < 4:
            continue
        cleaned.extend(terminal_dt_variants(normalized))
    return sorted(set(cleaned), key=len, reverse=True)


def alias_token_terms(aliases: list[str]) -> set[str]:
    tokens: set[str] = set()
    for alias in aliases:
        normalized_alias = normalize_match(alias)
        for match in _WORD_RE.finditer(normalized_alias):
            token = match.group(0)
            if token not in _ALIAS_TOKEN_STOP_TERMS:
                tokens.add(token)
    return tokens


def focused_question_terms(question: str, aliases: list[str], max_terms: int = 8) -> list[str]:
    alias_tokens = alias_token_terms(aliases)
    focused: list[str] = []
    for term in question_terms(question, max_terms=max_terms + len(alias_tokens) + 4):
        if term in alias_tokens or term in _ALIAS_TOKEN_STOP_TERMS:
            continue
        focused.append(term)
        if len(focused) >= max_terms:
            break
    return focused


def alias_hits(text: str, aliases: list[str]) -> int:
    normalized_text = normalize_match(text)
    hits = 0
    for alias in normalized_aliases(aliases):
        pattern = rf"\b{re.escape(alias)}\b"
        matches = len(re.findall(pattern, normalized_text))
        if not matches:
            continue
        # Multi-word aliases are safer evidence than a single first name.
        hits += matches * (3 if " " in alias else 1)
    return hits


def term_hits(text: str, terms: list[str]) -> int:
    normalized_text = normalize_match(text)
    hits = 0
    for term in terms:
        variants = terminal_dt_variants(term) or [term]
        if any(re.search(rf"\b{re.escape(variant)}\b", normalized_text) for variant in variants):
            hits += 1
    return hits


def score_text(
    text: str,
    aliases: list[str],
    terms: list[str],
    chunk_type: str = "source",
    require_alias: bool = False,
) -> float:
    alias_score = alias_hits(text, aliases)
    if require_alias and alias_score <= 0:
        return 0.0
    question_score = term_hits(text, terms)
    type_adjustment = -6 if chunk_type == "discussion_question" else 0
    return float(alias_score * 10 + question_score + type_adjustment)


def source_priority_score(figure_slug: str, row: dict) -> float:
    normalized_title = normalize_match(str(row.get("book_title") or ""))
    start_page = int(row.get("start_page") or 0)
    end_page = int(row.get("end_page") or 0)
    for pattern, page, boost in _FIGURE_SOURCE_PRIORITIES.get(figure_slug, ()):
        if pattern in normalized_title and (page is None or start_page <= page <= end_page):
            return float(boost)
    return 0.0


class FigureRetriever:
    def __init__(self, db: Database) -> None:
        self.db = db

    def retrieve(
        self,
        figure_id: int,
        question: str,
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        figure = self.db.get_figure(figure_id)
        if not figure:
            return []

        aliases = [figure["name"], *(figure.get("aliases") or [])]
        terms = focused_question_terms(question, aliases)
        rows = self.db.search_chunks(search_patterns(aliases, terms), limit=800)

        scored: list[RetrievedChunk] = []
        for row in rows:
            alias_score = alias_hits(row["text"], aliases)
            if alias_score <= 0:
                continue
            question_score = term_hits(row["text"], terms)
            priority_score = (
                source_priority_score(str(figure.get("slug") or ""), row)
                if not terms or question_score > 0
                else 0.0
            )
            if terms and question_score <= 0 and priority_score <= 0:
                continue
            score = score_text(
                row["text"],
                aliases=aliases,
                terms=terms,
                chunk_type=row.get("chunk_type", "source"),
                require_alias=True,
            )
            score += question_score * 12 + priority_score
            if score <= 0:
                continue
            scored.append(RetrievedChunk.model_validate({**row, "score": score}))

        if not scored and terms:
            rows = self.db.search_chunks(search_patterns(aliases, []), limit=300)
            for row in rows:
                score = score_text(
                    row["text"],
                    aliases=aliases,
                    terms=[],
                    chunk_type=row.get("chunk_type", "source"),
                    require_alias=True,
                )
                score += source_priority_score(str(figure.get("slug") or ""), row)
                if score <= 0:
                    continue
                scored.append(RetrievedChunk.model_validate({**row, "score": score}))

        return sorted(scored, key=lambda chunk: chunk.score, reverse=True)[:limit]

    def retrieve_general(self, question: str, limit: int = 5) -> list[RetrievedChunk]:
        terms = question_terms(question, max_terms=10)
        rows = self.db.search_chunks(search_patterns([], terms), limit=90)

        scored: list[RetrievedChunk] = []
        for row in rows:
            score = score_text(
                row["text"],
                aliases=[],
                terms=terms,
                chunk_type=row.get("chunk_type", "source"),
            )
            if score <= 0:
                continue
            scored.append(RetrievedChunk.model_validate({**row, "score": score}))

        return sorted(scored, key=lambda chunk: chunk.score, reverse=True)[:limit]


def retrieve_chunks_for_figure(db: Database, figure_id: int, question: str, limit: int = 5) -> list[RetrievedChunk]:
    retriever = FigureRetriever(db)
    return retriever.retrieve(figure_id, question, limit=limit)

def judge_retrieval(
    question: str,
    retrieved_context: str,
) -> dict:
    return judge_answer(
        question=(
            "Aşağıdaki chunk'lar kullanıcının asıl sorusunu "
            f"cevaplamak için yeterli mi?\n\nAsıl soru: {question}"
        ),
        context=retrieved_context,
        reference_answer=(
            "Seçilen metinlerin soruyu cevaplamak için gerekli "
            "bilgileri içermesi beklenmektedir."
        ),
        answer=(
            "Retriever tarafından seçilen kaynaklar bunlardır."
        ),
    )

