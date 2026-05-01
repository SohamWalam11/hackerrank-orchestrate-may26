"""
Retriever module — loads the text corpus, chunks documents, and performs
hybrid BM25 + TF-IDF retrieval with Reciprocal Rank Fusion (RRF).

Design decisions:
  • BM25 (via rank_bm25) for keyword-aware, length-normalized ranking.
  • TF-IDF (via sklearn) as a complementary signal for n-gram similarity.
  • RRF merges both rankings without needing tuned score calibration.
  • Sentence-aware chunking avoids splitting mid-sentence where possible.
  • Company-filtered retrieval with fallback to full corpus.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

np.random.seed(config.SEED)

# Extensions we know how to parse
_SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".json"}


# ---------------------------------------------------------------------------
# File reading helpers
# ---------------------------------------------------------------------------

def _read_file(filepath: str) -> str:
    """Read a file and return its text content, stripping format markers."""
    ext = Path(filepath).suffix.lower()
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
    except Exception:
        return ""

    if ext == ".html":
        raw = re.sub(r"<[^>]+>", " ", raw)
        raw = re.sub(r"\s+", " ", raw).strip()
    elif ext == ".json":
        try:
            data = json.loads(raw)
            strings = _extract_strings(data)
            raw = " ".join(strings)
        except json.JSONDecodeError:
            raw = ""
    elif ext == ".md":
        # Strip markdown headers/bold/links but keep text
        raw = re.sub(r"!\[.*?\]\(.*?\)", " ", raw)       # images
        raw = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", raw)  # links → text
        raw = re.sub(r"#{1,6}\s*", "", raw)               # headers
        raw = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", raw)  # bold/italic
    return raw.strip()


def _extract_strings(obj) -> List[str]:
    """Recursively extract all string values from a JSON-like structure."""
    results = []
    if isinstance(obj, str):
        results.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            results.extend(_extract_strings(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            results.extend(_extract_strings(item))
    return results


def _infer_company_tag(filepath: str, corpus_root: str) -> str:
    """
    Infer company tag from directory structure.
    e.g. data/hackerrank/... → "hackerrank"
    """
    rel = os.path.relpath(filepath, corpus_root)
    parts = Path(rel).parts
    if len(parts) > 1:
        return parts[0].lower()
    return "unknown"


def _infer_topic_from_path(filepath: str, corpus_root: str) -> str:
    """Extract a topic hint from the file path for metadata enrichment."""
    rel = os.path.relpath(filepath, corpus_root)
    parts = Path(rel).parts
    # Skip company name (parts[0]), use remaining directory names
    topics = [p.replace("-", " ").replace("_", " ") for p in parts[1:-1]]
    return " > ".join(topics) if topics else ""


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _sentence_aware_chunk(text: str, source: str, company_tag: str, topic: str) -> List[Dict]:
    """
    Split text into overlapping chunks, breaking at sentence boundaries
    where possible. Returns list of dicts with text, source, company_tag, topic.
    """
    if not text.strip():
        return []

    chunks = []
    chunk_size = config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP
    step = chunk_size - overlap

    # Split into sentences for cleaner boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)

    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            if len(current_chunk.strip()) >= config.MIN_CHUNK_LENGTH:
                chunks.append({
                    "text": current_chunk.strip(),
                    "source": source,
                    "company_tag": company_tag,
                    "topic": topic,
                })
            # Keep overlap from end of current chunk
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + " " + sentence
        else:
            current_chunk += (" " if current_chunk else "") + sentence

    # Don't forget the last chunk
    if current_chunk.strip() and len(current_chunk.strip()) >= config.MIN_CHUNK_LENGTH:
        chunks.append({
            "text": current_chunk.strip(),
            "source": source,
            "company_tag": company_tag,
            "topic": topic,
        })

    # Fallback: if sentence splitting produced nothing, use character-based
    if not chunks and len(text.strip()) >= config.MIN_CHUNK_LENGTH:
        for i in range(0, max(1, len(text)), step):
            segment = text[i:i + chunk_size]
            if segment.strip() and len(segment.strip()) >= config.MIN_CHUNK_LENGTH:
                chunks.append({
                    "text": segment.strip(),
                    "source": source,
                    "company_tag": company_tag,
                    "topic": topic,
                })

    return chunks


# ---------------------------------------------------------------------------
# Main retriever class
# ---------------------------------------------------------------------------

class CorpusRetriever:
    """Hybrid BM25 + TF-IDF retriever over a local text corpus."""

    def __init__(self, corpus_root: Optional[str] = None):
        self.corpus_root = corpus_root or str(config.CORPUS_ROOT)
        self.chunks: List[Dict] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.bm25 = None
        self.tokenized_chunks: List[List[str]] = []
        self._load_corpus()

    def _load_corpus(self):
        """Walk the corpus directory, read files, chunk them, build indices."""
        raw_chunks = []
        for dirpath, _, filenames in os.walk(self.corpus_root):
            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                if Path(fname).suffix.lower() not in _SUPPORTED_EXTENSIONS:
                    continue
                text = _read_file(fpath)
                if not text:
                    continue
                company = _infer_company_tag(fpath, self.corpus_root)
                topic = _infer_topic_from_path(fpath, self.corpus_root)
                raw_chunks.extend(
                    _sentence_aware_chunk(text, fpath, company, topic)
                )

        self.chunks = raw_chunks
        if not self.chunks:
            return

        texts = [c["text"] for c in self.chunks]

        # Build TF-IDF index
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=15000,
            sublinear_tf=True,    # dampens term frequency
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

        # Build BM25 index
        if HAS_BM25:
            self.tokenized_chunks = [
                self._tokenize(t) for t in texts
            ]
            self.bm25 = BM25Okapi(self.tokenized_chunks)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + lowercase tokenizer for BM25."""
        return re.findall(r'\w+', text.lower())

    def retrieve(
        self,
        query: str,
        company: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Retrieve top-k chunks most similar to the query using hybrid ranking.

        If company is given, filter chunks to that company first.
        Falls back to full corpus when the filter yields too few results.
        Uses Reciprocal Rank Fusion (RRF) when BM25 is available.
        """
        if self.vectorizer is None or self.tfidf_matrix is None:
            return []

        top_k = top_k or config.TOP_K_CHUNKS

        # Determine which chunk indices to search over
        indices = list(range(len(self.chunks)))
        if company:
            company_lower = company.lower()
            filtered = [
                i for i, c in enumerate(self.chunks)
                if c["company_tag"] == company_lower
            ]
            if len(filtered) >= 3:
                indices = filtered

        # --- TF-IDF scoring ---
        query_vec = self.vectorizer.transform([query])
        tfidf_sims = cosine_similarity(query_vec, self.tfidf_matrix[indices])[0]

        # --- BM25 scoring (if available) ---
        if self.bm25 and HAS_BM25:
            query_tokens = self._tokenize(query)
            # BM25 scores over full corpus, then pick relevant indices
            full_bm25_scores = self.bm25.get_scores(query_tokens)
            bm25_scores = np.array([full_bm25_scores[i] for i in indices])

            # RRF: fuse rankings
            rrf_scores = self._rrf_fusion(tfidf_sims, bm25_scores)
            top_local = np.argsort(rrf_scores)[::-1][:top_k]
        else:
            top_local = np.argsort(tfidf_sims)[::-1][:top_k]

        results = []
        for idx in top_local:
            global_idx = indices[idx]
            # Compute a combined score for display
            score = float(tfidf_sims[idx])
            if score > 0:
                results.append({
                    "text": self.chunks[global_idx]["text"],
                    "source": self.chunks[global_idx]["source"],
                    "topic": self.chunks[global_idx].get("topic", ""),
                    "company_tag": self.chunks[global_idx]["company_tag"],
                    "score": round(score, 4),
                })

        return results

    @staticmethod
    def _rrf_fusion(
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        k: int = 60,
    ) -> np.ndarray:
        """
        Reciprocal Rank Fusion of two score arrays.
        RRF(d) = sum_r 1/(k + rank_r(d))
        """
        n = len(scores_a)
        rank_a = np.argsort(np.argsort(-scores_a))  # 0 = best
        rank_b = np.argsort(np.argsort(-scores_b))

        rrf = np.zeros(n)
        for i in range(n):
            rrf[i] = (
                config.TFIDF_WEIGHT / (k + rank_a[i])
                + config.BM25_WEIGHT / (k + rank_b[i])
            )
        return rrf
