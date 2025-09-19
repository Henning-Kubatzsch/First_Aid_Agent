# src/rag/chunk.py
from __future__ import annotations
import re
import hashlib
from typing import List, Dict, Any, Optional
from rag.interfaces import Chunker


class ParagraphChunker(Chunker):
    """
    Simple, deterministic paragraph splitter with approximate size control.
    - Splits on blank lines (paragraphs)
    - Merges paragraphs until target_chars (but ensures >= min_chars)
    - Adds character overlap *between* emitted chunks (never before the first)
    """

    ## def __init__(self, target_chars: int = 1200, min_chars: int = 400, overlap_chars: int = 120):
    def __init__(self, target_chars: int = 400, min_chars: int = 200, overlap_chars: int = 120):
        self.target = target_chars
        self.min = min_chars
        self.overlap = overlap_chars

    def _norm(self, txt: str) -> str:
        txt = txt.replace("\r", "")
        txt = re.sub(r'\n{2,}', '\n\n', txt)
        return txt.strip()


    def split(self, text: str, meta: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        meta = meta or {}
        text = self._norm(text)

        # sections are separated by "---"
        paras = [p.strip() for p in re.split(r"---", text) if p.strip()]

        # now i have a list of sections

        chunks: List[Dict[str, Any]] = []
       

        # TODO: 
        # - iterate through paras 
        # - iterate through questions + embed them
        # - for every question do embedding individually and add them to chunks

    
        cid = hashlib.sha1(chunk_text.encode("utf-8")).hexdigest()[:16]
        out = {"id": cid, "text": chunk_text, "meta": dict(meta)}
        chunks.append(out)




     
        return chunks
