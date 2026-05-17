import os
import re
import json
import bm25s
import yaml
from pathlib import Path
from langchain_core.messages import SystemMessage


_YOUTUBE_ID_RE = re.compile(
    r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/|youtube\.com/shorts/)([\w-]{11})'
)


def extract_youtube_id(url: str) -> str | None:
    """Pull the 11-char video ID from any common YouTube URL form, or accept a bare ID."""
    m = _YOUTUBE_ID_RE.search(url)
    if m:
        return m.group(1)
    if re.fullmatch(r'[\w-]{11}', url.strip()):
        return url.strip()
    return None


_FINAL_ANSWER_RE = re.compile(r'FINAL ANSWER:\s*(.*)', re.DOTALL | re.IGNORECASE)


def extract_final_answer(content: str) -> str:
    """Pull the value after 'FINAL ANSWER:' (case-insensitive), or return the content stripped."""
    content = content or ""
    m = _FINAL_ANSWER_RE.search(content)
    return (m.group(1) if m else content).strip()


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_prompt(prompt_location: str) -> SystemMessage:
    """Load system prompt from YAML file."""
    with open(prompt_location) as f:
        try:
            prompt = yaml.safe_load(f)["prompt"]
            return SystemMessage(content=prompt)
        except yaml.YAMLError as exc:
            print(exc)
            return SystemMessage(content="You are a helpful assistant.")



def init_bm25_index(corpus_file = "data/metadata.jsonl"):
    """BM25 Index Initialization (Local Corpus)"""
    try:
        if not os.path.exists(corpus_file):
            print(f"Warning: {corpus_file} not found. BM25 will use empty index.")
            return None, [], []
            
        search_texts = []  # question-only — used for BM25 indexing
        corpus_texts = []  # Q+A+Steps — returned for context injection
        corpus_ids = []
        with open(corpus_file, "r") as f:
            for line in f:
                item = json.loads(line)
                question = item.get('Question', '')
                answer = item.get('Final answer', '')
                steps = item.get('Annotator Metadata', {}).get('Steps', '')
                search_texts.append(question)
                parts = [f"Question: {question}"]
                if answer:
                    parts.append(f"Final Answer: {answer}")
                if steps:
                    parts.append(f"Solution Steps: {steps}")
                corpus_texts.append("\n".join(parts))
                corpus_ids.append(item.get('task_id', ''))

        corpus_tokens = bm25s.tokenize(search_texts, stopwords="en", stemmer=None)
        
        retriever_bm25 = bm25s.BM25()
        retriever_bm25.index(corpus_tokens)
        
        print(f"BM25 Index initialized with {len(corpus_texts)} documents.")
        return retriever_bm25, corpus_texts, corpus_ids
    except Exception as e:
        print(f"Error initializing BM25: {e}")
        return None, [], []
    

def reciprocal_rank_fusion(results: list[list[dict]], k=60) -> list[tuple[dict, float]]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).
    """
    fused_scores = {}
    
    for rank_list in results:
        for rank, doc in enumerate(rank_list):
            doc_id = doc["metadata"]["task_id"]
            doc_content = doc["content"]
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"id": doc_id, "content": doc_content, "score": 0.0}
            fused_scores[doc_id]["score"] += 1.0 / (k + rank + 1)
            
    sorted_results = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
    return [(item["id"], item["content"], item["score"]) for item in sorted_results]