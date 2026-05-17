"""
GAIA Agent with Multi-Modal File Processing and Hybrid Retrieval.

This module defines a LangGraph agent that can:
1. Retrieve similar questions using Hybrid Search (Vector + BM25) and Reranking
2. Process files using tools (PDF, XLSX, MP3, etc.)
3. Answer questions using web search, calculator, and other tools
"""

import os
import bm25s
from pathlib import Path
from dotenv import load_dotenv

from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode

from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from supabase.client import Client, create_client

from gaia.utils import (
    load_config, load_prompt, init_bm25_index, reciprocal_rank_fusion,
    extract_final_answer, download_task_file,
)
from gaia.tools import tools_list
from gaia.states import AgentState

load_dotenv()
config = load_config()

# Environment details and others
hf_key = os.getenv("HF_INFERENCE_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

# ============================================
# Model & Embeddings Setup
# ============================================

enable_keyword_search = config["retrievers"]["enable_keyword_search"]
enable_vector_search = config["retrievers"]["enable_vector_search"]

# BM25 Retriever
bm25_retriever, bm25_corpus, bm25_ids = None, None, None
if enable_keyword_search:
    bm25_retriever, bm25_corpus, bm25_ids = init_bm25_index(corpus_file=config["data"])

bm25_id_to_text = {}
if bm25_corpus and bm25_ids:
    bm25_id_to_text = dict(zip(bm25_ids, bm25_corpus))

embeddings, supabase = None, None
if enable_vector_search:
    # Embeddings for Vector Search
    embeddings = SentenceTransformer(model_name_or_path=config["models"]["embeddings"]["model_name"], cache_folder=config["models"]["cache_folder"])

    # Supabase Vector Store
    supabase: Client = create_client(supabase_url, supabase_key)

# Reranker Model (ModernBERT Cross-Encoder)
reranker = CrossEncoder(config["models"]["reranker"]["model_name"], cache_folder=config["models"]["cache_folder"])

# LLM for Agent
llm = HuggingFaceEndpoint(
    repo_id=config["models"]["llm"]["model_name"],
    temperature=config["models"]["llm"]["parameters"]["temperature"],
    repetition_penalty=config["models"]["llm"]["parameters"]["repetition_penalty"],
    provider=config["models"]["llm"]["parameters"]["provider"],
    huggingfacehub_api_token=hf_key
)

agent_llm = ChatHuggingFace(llm=llm)
agent_with_tools = agent_llm.bind_tools(tools_list)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_system_prompt = load_prompt(str(_PROMPTS_DIR / "prompt.yaml"))
_thinking_enabled = config["models"]["llm"]["parameters"].get("thinking_enabled", True)


@tool
def emit_final_answer(answer: str) -> str:
    """Emit the final answer to the GAIA question in the strict scoring format.

    Args:
        answer: The raw answer value only.
            Numbers: plain digits, no commas, no units, no symbols (write '1000000', not '1,000,000' or '$50').
            Strings: no articles ('a', 'an', 'the'), no markdown, no surrounding quotes, no trailing punctuation.
            Lists: comma-separated with no extra spaces, in the order requested by the question.
    """
    return answer


formatter_llm = agent_llm.bind_tools([emit_final_answer], tool_choice="emit_final_answer")


# ============================================
# Graph Nodes
# ============================================

def file_downloader_node(state: AgentState) -> AgentState:
    """
    Download the task file from the scoring API if one is associated with the question.
    Saves to a local directory and stores the path in state. Detailed failure
    diagnostics are logged so HF Space logs reveal the actual cause (network, HTTP, filesystem).
    """
    print("--- FILE DOWNLOADER NODE ---")
    file_name = state.get("file_name", "")
    task_id = state.get("task_id", "")

    if not file_name or not task_id:
        return {"file_path": ""}

    local_path, err = download_task_file(
        task_id=task_id,
        file_name=file_name,
        base_url=config["api"]["base_url"],
        files_dir=config["api"]["files_dir"],
    )
    if local_path:
        print(f"Downloaded: {file_name} → {local_path}")
        return {"file_path": local_path}
    print(f"File download failed for task {task_id} ({file_name}): {err}")
    return {"file_path": ""}


def retriever_node(state: AgentState) -> AgentState:
    """
    Hybrid Search Node: Retrieve docs via Vector Search + BM25, combine with RRF.
    """
    print("--- RETRIEVER NODE ---")
    messages = state.get("messages", [])
    if not messages:
        return {"retrieved_docs": []}
    
    question_content = messages[0].content
    
    if not enable_vector_search and not enable_keyword_search:
        print("No retrieval method enabled.")
        return {"retrieved_docs": []}
    
    # 1. Vector Search
    vector_docs = []
    if supabase and embeddings:
        try:
            response = supabase.rpc(
                config["retrievers"]["vector_store"]["query"],
                {"query_embedding": embeddings.encode(question_content).tolist(), 
                 "match_count": config["retrievers"]["vector_store"]["k"], 
                 "match_threshold": config["retrievers"]["vector_store"]["threshold"]
                 }     
            ).execute()

            vector_docs = response.data

        except Exception as e:
            print(f"Vector search error: {e}")
        
    # 2. BM25 Search
    bm25_docs = []
    if bm25_retriever and bm25_corpus and bm25_ids:
        try:
            query_tokens = bm25s.tokenize([question_content], stopwords="en")
            results, scores = bm25_retriever.retrieve(query_tokens, k=config["retrievers"]["bm25"]["k"])
            indices = results[0]
            
            for i, idx in enumerate(indices):
                content = bm25_corpus[idx]
                task_id = bm25_ids[idx]
                score = scores[0][i]
                bm25_dict = {"content":content, "metadata": {"source": "bm25_search", "task_id": task_id, "score": score}}
                bm25_docs.append(bm25_dict)
        except Exception as e:
            print(f"BM25 search error: {e}")

    # 3. RRF Fusion
    final_candidates = []
    if vector_docs and bm25_docs:
        fused = reciprocal_rank_fusion([vector_docs, bm25_docs])
        final_candidates = [id for id, doc, score in fused]
    else:
        final_candidates = vector_docs + bm25_docs
        final_candidates = [doc["metadata"]["task_id"] for doc in final_candidates]
        
    top_candidates = final_candidates[:20]
    
    return {"retrieved_docs": top_candidates}


def reranker_node(state: AgentState) -> AgentState:
    """
    Reranker Node: Re-order candidates using Cross-Encoder and return top 3.
    """
    print("--- RERANKER NODE ---")
    candidates = state.get("retrieved_docs", [])
    messages = state.get("messages", [])
    
    if not candidates or not messages:
        return {"messages": []}
        
    question = messages[0].content
    
    # Deduplicate candidates — candidates are task_id strings; resolve to text via corpus lookup
    unique_candidates = []
    seen_content = set()
    for task_id in candidates:
        text = bm25_id_to_text.get(task_id)
        if text and text not in seen_content:
            unique_candidates.append(text)
            seen_content.add(text)
            
    if not unique_candidates:
        return {"messages": []}

    pairs = [[question, doc_text] for doc_text in unique_candidates]
    
    try:
        scores = reranker.predict(pairs)
        
        scored_docs = sorted(
            zip(unique_candidates, scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        top_k = config["retrievers"]["final_rrf_k"]
        top_results = scored_docs[:top_k]
        
        context_str = "Here are similar questions and answers for reference:\n\n"
        for i, (doc_text, score) in enumerate(top_results):
            context_str += f"--- Example {i+1} (Score: {score:.2f}) ---\n{doc_text}\n\n"
            
        context_message = HumanMessage(content=context_str)
        
        return {"messages": [context_message]}
        
    except Exception as e:
        print(f"Reranker error: {e}")
        if unique_candidates:
             fallback_msg = HumanMessage(content=f"Reference (Fallback):\n{unique_candidates[0]}")
             return {"messages": [fallback_msg]}
             
    return {"messages": []}


def processor_node(state: AgentState) -> AgentState:
    """
    Processor Node: Main LLM agent that answers the question.
    """
    prompt_content = _system_prompt.content + ("" if _thinking_enabled else "\n/no_think")
    system_prompt = SystemMessage(content=prompt_content)
    messages = state.get("messages", [])
    task_id = state.get("task_id", "")
    file_name = state.get("file_name", "")
    file_path = state.get("file_path", "")

    full_messages = [system_prompt]

    if file_name:
        if file_path:
            file_msg = HumanMessage(
                content=f"Note: A file named '{file_name}' is associated with this question. It is available at path: {file_path}"
            )
        else:
            file_msg = HumanMessage(
                content=(
                    f"Note: A file named '{file_name}' is associated with this question "
                    f"(task_id={task_id}), but the automatic download failed. "
                    f"Call `retry_file_download(task_id='{task_id}', file_name='{file_name}')` "
                    f"once to attempt it again. On success, you'll get back the local path you "
                    f"can pass to the appropriate read tool. If retry also fails, state clearly "
                    f"that the file is unavailable rather than fabricating data."
                )
            )
        full_messages.append(file_msg)
    
    full_messages.extend(messages)
    
    response = agent_with_tools.invoke(full_messages)

    return {"messages": [response]}


def formatter_node(state: AgentState) -> AgentState:
    """Extract and reformat the solver's answer into a strict GAIA-compliant value."""
    print("--- FORMATTER NODE ---")
    messages = state.get("messages", [])
    if not messages:
        return {"final_answer": ""}

    question = ""
    for m in messages:
        if isinstance(m, HumanMessage):
            question = m.content
            break

    solver_output = messages[-1].content or ""

    prompt = [
        SystemMessage(content=(
            "You extract the final answer from an agent's reasoning and apply the GAIA "
            "formatting rules exactly. You MUST call the `emit_final_answer` tool with "
            "the extracted value. If the agent never produced an answer, call it with an "
            "empty string."
        )),
        HumanMessage(content=(
            f"Question:\n{question}\n\n"
            f"Agent reasoning and conclusion:\n{solver_output}\n\n"
            "Extract the final answer and call emit_final_answer."
        )),
    ]

    try:
        result = formatter_llm.invoke(prompt)
        for tc in getattr(result, "tool_calls", None) or []:
            if tc.get("name") == "emit_final_answer":
                return {"final_answer": str(tc.get("args", {}).get("answer", "")).strip()}
        # Model returned text instead of calling the tool — fall back to regex.
        return {"final_answer": extract_final_answer(result.content)}
    except Exception as e:
        print(f"Formatter error: {e}")
        return {"final_answer": extract_final_answer(solver_output)}


# ============================================
# Graph Construction
# ============================================

def agent_graph():
    """
    Build and compile the agent graph.
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("file_downloader_node", file_downloader_node)
    workflow.add_node("retriever_node", retriever_node)
    workflow.add_node("reranker_node", reranker_node)
    workflow.add_node("processor_node", processor_node)
    workflow.add_node("tools", ToolNode(tools_list))
    workflow.add_node("formatter_node", formatter_node)

    # Add edges
    workflow.add_edge(START, "file_downloader_node")
    workflow.add_edge("file_downloader_node", "retriever_node")
    workflow.add_edge("retriever_node", "reranker_node")
    workflow.add_edge("reranker_node", "processor_node")
    workflow.add_edge("tools", "processor_node")
    workflow.add_conditional_edges(
        "processor_node",
        tools_condition,
        {"tools": "tools", END: "formatter_node"},
    )
    workflow.add_edge("formatter_node", END)
    
    
    compiled = workflow.compile()
    return compiled.with_config({"recursion_limit": config["graph"]["recursion_limit"]})
