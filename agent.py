"""
GAIA Agent with Multi-Modal File Processing and Hybrid Retrieval.

This module defines a LangGraph agent that can:
1. Retrieve similar questions using Hybrid Search (Vector + BM25) and Reranking
2. Process files using tools (PDF, XLSX, MP3, etc.)
3. Answer questions using web search, calculator, and other tools
"""

import os
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, BaseMessage
from supabase.client import Client, create_client
from sentence_transformers import CrossEncoder

from utils import load_prompt, init_bm25_index, reciprocal_rank_fusion
from tools import tools_list

load_dotenv()


# ============================================
# State Definition
# ============================================

class AgentState(TypedDict):
    """
    State schema for the GAIA agent graph.
    
    Attributes:
        messages: List of conversation messages (auto-accumulated via add_messages)
        task_id: The GAIA task identifier for the current question
        file_name: Name of the attached file (empty string if no file)
        retrieved_docs: List of candidate documents from the retriever node
    """
    messages: Annotated[list[BaseMessage], add_messages]
    task_id: str
    file_name: str
    retrieved_docs: List[Document]


# ============================================
# Model & Embeddings Setup
# ============================================

# Environment details and others
hf_key = os.getenv("HF_INFERENCE_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
supabase_table = "gaia_documents"
supabase_query = "match_documents_langchain"

# Model List
embeddings_model_name = "Alibaba-NLP/gte-modernbert-base"
reranker_model_name = "Alibaba-NLP/gte-reranker-modernbert-base"
llm_model_name = "Qwen/Qwen3-32B-Instruct"

# Embeddings for Vector Search
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

# Supabase Vector Store
supabase: Client = create_client(supabase_url, supabase_key)

vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name=supabase_table,
    query_name=supabase_query,
)

# BM25 Retriever
bm25_retriever, bm25_corpus = init_bm25_index()

# Reranker Model (ModernBERT Cross-Encoder)
reranker = CrossEncoder(reranker_model_name)

# LLM Setup

llm = HuggingFaceEndpoint(
    repo_id=llm_model_name,
    temperature=0,
    repetition_penalty=1.03,
    provider="auto",
    huggingfacehub_api_token=hf_key
)

agent_llm = ChatHuggingFace(llm=llm)
agent_with_tools = agent_llm.bind_tools(tools_list)


# ============================================
# Graph Nodes
# ============================================

def retriever_node(state: AgentState) -> AgentState:
    """
    Hybrid Search Node: Retrieve docs via Vector Search + BM25, combine with RRF.
    """
    print("--- RETRIEVER NODE ---")
    messages = state.get("messages", [])
    if not messages:
        return {"retrieved_docs": []}
    
    question_content = messages[0].content
    
    # 1. Vector Search
    vector_docs = []
    try:
        vector_docs = vector_store.similarity_search(question_content, k=10)
    except Exception as e:
        print(f"Vector search error: {e}")
        
    # 2. BM25 Search
    bm25_docs = []
    if bm25_retriever and bm25_corpus:
        try:
            query_tokens = bm25s.tokenize([question_content], stopwords="en")
            results, scores = bm25_retriever.retrieve(query_tokens, k=10)
            indices = results[0]
            
            for idx in indices:
                doc_content = bm25_corpus[idx]
                bm25_docs.append(Document(page_content=doc_content, metadata={"source": "bm25"}))
        except Exception as e:
            print(f"BM25 search error: {e}")

    # 3. RRF Fusion
    final_candidates = []
    if vector_docs and bm25_docs:
        fused = reciprocal_rank_fusion([vector_docs, bm25_docs])
        final_candidates = [doc for doc, score in fused]
    else:
        final_candidates = vector_docs + bm25_docs
        
    top_candidates = final_candidates[:20]
    
    return {"retrieved_docs": top_candidates}


def reranker_node(state: AgentState) -> AgentState:
    """
    Reranker Node: Re-order candidates using ModernBERT Cross-Encoder and return top 3.
    """
    print("--- RERANKER NODE ---")
    candidates = state.get("retrieved_docs", [])
    messages = state.get("messages", [])
    
    if not candidates or not messages:
        return {"messages": []}
        
    question = messages[0].content
    
    # Deduplicate candidates
    unique_candidates = []
    seen_content = set()
    for doc in candidates:
        if doc.page_content not in seen_content:
            unique_candidates.append(doc.page_content)
            seen_content.add(doc.page_content)
            
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
        
        top_k = 3
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
    system_prompt = load_prompt("prompts/prompt.yaml")
    messages = state.get("messages", [])
    file_name = state.get("file_name", "")
    task_id = state.get("task_id", "")
    
    full_messages = [system_prompt]
    
    if file_name:
        file_msg = HumanMessage(
            content=f"Note: A file named '{file_name}' is associated with this question (task_id: {task_id})."
        )
        full_messages.append(file_msg)
    
    full_messages.extend(messages)
    
    response = agent_with_tools.invoke(full_messages)
    
    return {"messages": [response]}


# ============================================
# Graph Construction
# ============================================

def agent_graph():
    """
    Build and compile the agent graph.
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("retriever_node", retriever_node)
    workflow.add_node("reranker_node", reranker_node)
    workflow.add_node("processor_node", processor_node)
    workflow.add_node("tools", ToolNode(tools_list))

    # Add edges
    workflow.add_edge(START, "retriever_node")
    workflow.add_edge("retriever_node", "reranker_node")
    workflow.add_edge("reranker_node", "processor_node")
    workflow.add_edge("tools", "processor_node")    
    workflow.add_conditional_edges("processor_node", tools_condition, {"tools": "tools", END: END})
    
    
    return workflow.compile()
