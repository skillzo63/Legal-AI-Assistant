"""Tests for legal-structural chunker."""

from rag.chunker import LegalChunk, LegalStructuralChunker


def test_chunker_basic():
    chunker = LegalStructuralChunker(max_chars=300, min_chars=50)
    source = {
        "citation": "Test Case [2024] HCA 1",
        "url": "https://example.com/test",
        "jurisdiction": "high_court",
        "type": "decision",
        "text": (
            " 1 The plaintiff claims that the defendant was negligent. "
            "The incident occurred on 12 March 2023 at the premises. "
            " 2 The defendant denies all liability and asserts contributory negligence. "
            "The court heard arguments from both senior counsel over two days."
        ),
    }

    chunks = chunker.chunk_document(doc_id=42, source_dict=source, start_chunk_id=100)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert isinstance(chunk, LegalChunk)
        assert chunk.doc_id == 42
        assert chunk.citation == "Test Case [2024] HCA 1"
        assert chunk.url == "https://example.com/test"
        assert "Test Case [2024] HCA 1" in chunk.injected_text
        assert "high_court" in chunk.injected_text
        assert len(chunk.clean_text) > 0


def test_chunker_empty():
    chunker = LegalStructuralChunker()
    chunks = chunker.chunk_document(doc_id=1, source_dict={"text": ""})
    assert len(chunks) == 0


def test_chunker_para_markers():
    chunker = LegalStructuralChunker()
    paras = chunker.extract_para_markers(" 3 The plaintiff... [29] The evidence... Section 15...")
    assert "3" in paras
    assert "29" in paras
    assert "15" in paras
