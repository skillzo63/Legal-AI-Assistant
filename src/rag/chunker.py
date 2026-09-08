"""Legal-structural and contextual chunker for Australian case law and statutes."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field


@dataclass
class LegalChunk:
    """Represents a discrete semantic chunk of a legal document."""

    chunk_id: int
    doc_id: int
    citation: str
    url: str
    jurisdiction: str
    doc_type: str
    para_markers: list[str] = field(default_factory=list)
    clean_text: str = ""
    injected_text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> LegalChunk:
        return cls(**data)


class LegalStructuralChunker:
    """Splits legal judgments and statutes on structural boundaries with context injection."""

    # Matches paragraph markers (e.g. ' 3 The', '4 This', '[29] The', '1. Gift')
    # and sentence boundaries following punctuation.
    SPLIT_REGEX = re.compile(
        r"(?:\n+|\r+|\s{2,}|\s(?=\[?\d+\]?\.\s+[A-Z])|\s(?=\[?\d+\]?\s+[A-Z])"
        r"|(?<=[.!?])\s+(?=[A-Z]))"
    )

    # Identifies paragraph/section numbers inside a chunk
    PARA_EXTRACTOR = re.compile(
        r"(?:^|\s)(?:\[(\d+)\]|(\d+)\.|\bSection\s+(\d+)|\b(?:para|paragraph)\s+(\d+)|(\d+)\b)"
    )

    def __init__(
        self,
        max_chars: int = 1000,
        min_chars: int = 150,
        overlap_chars: int = 100,
    ) -> None:
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.overlap_chars = overlap_chars

    def split_raw_text(self, text: str) -> list[str]:
        """Split raw text into structurally coherent blocks within character limits."""
        text = text.strip()
        if not text:
            return []

        if len(text) <= self.max_chars:
            return [text]

        raw_segments = [s.strip() for s in self.SPLIT_REGEX.split(text) if s.strip()]
        if not raw_segments:
            return [text]

        chunks: list[str] = []
        current: list[str] = []
        curr_len = 0

        for seg in raw_segments:
            seg_len = len(seg)
            # If adding this segment exceeds max_chars and we already have text
            if curr_len + seg_len + 1 > self.max_chars and current:
                chunk_str = " ".join(current).strip()
                chunks.append(chunk_str)
                # Keep small overlap from previous segment if configured
                if self.overlap_chars > 0 and len(current[-1]) <= self.overlap_chars:
                    current = [current[-1], seg]
                    curr_len = len(current[0]) + seg_len + 1
                else:
                    current = [seg]
                    curr_len = seg_len
            else:
                current.append(seg)
                curr_len += seg_len + 1

        if current:
            chunk_str = " ".join(current).strip()
            chunks.append(chunk_str)

        # Merge trailing tiny chunks with the previous chunk if within budget
        merged: list[str] = []
        for c in chunks:
            if (
                merged
                and len(merged[-1]) + len(c) + 1 <= self.max_chars
                and len(c) < self.min_chars
            ):
                merged[-1] = merged[-1] + " " + c
            else:
                merged.append(c)

        return merged or [text]

    def extract_para_markers(self, text: str) -> list[str]:
        """Extract paragraph or section markers found in the chunk text."""
        markers = []
        for m in self.PARA_EXTRACTOR.finditer(text):
            val = next((g for g in m.groups() if g is not None), None)
            if val and val not in markers:
                markers.append(val)
        return markers

    def build_context_header(
        self,
        citation: str,
        jurisdiction: str,
        doc_type: str,
        para_markers: list[str],
    ) -> str:
        """Construct the contextual metadata header to prepend to the chunk."""
        parts = []
        if citation:
            parts.append(f"Citation: {citation}")
        if jurisdiction:
            parts.append(f"Jurisdiction: {jurisdiction}")
        if doc_type:
            parts.append(f"Type: {doc_type}")
        if para_markers:
            parts.append(f"Para: {', '.join(para_markers[:3])}")

        header_content = " | ".join(parts)
        return f"[{header_content}]" if header_content else ""

    def chunk_document(
        self,
        doc_id: int,
        source_dict: dict,
        start_chunk_id: int = 0,
    ) -> list[LegalChunk]:
        """Convert a document's source record into contextual legal chunks."""
        text = source_dict.get("text", "") or ""
        citation = source_dict.get("citation", "") or ""
        url = source_dict.get("url", "") or ""
        jurisdiction = source_dict.get("jurisdiction", "") or ""
        doc_type = source_dict.get("type", "") or ""

        text_blocks = self.split_raw_text(text)
        chunks: list[LegalChunk] = []

        for offset, block in enumerate(text_blocks):
            cid = start_chunk_id + offset
            paras = self.extract_para_markers(block)
            header = self.build_context_header(citation, jurisdiction, doc_type, paras)
            injected = f"{header}\n{block}".strip() if header else block

            chunks.append(
                LegalChunk(
                    chunk_id=cid,
                    doc_id=doc_id,
                    citation=citation,
                    url=url,
                    jurisdiction=jurisdiction,
                    doc_type=doc_type,
                    para_markers=paras,
                    clean_text=block,
                    injected_text=injected,
                )
            )

        return chunks
