"""
GAIA Agent with Multi-Modal File Processing.

This module defines a LangGraph agent that can:
1. Retrieve similar questions from a vector store (RAG)
2. Process files using tools (PDF, XLSX, MP3, etc.)
3. Answer questions using web search, calculator, and other tools
"""

import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.messages import HumanMessage, BaseMessage
from supabase.client import Client, create_client

from utils import load_prompt
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
    """
    messages: Annotated[list[BaseMessage], add_messages]
    task_id: str
    file_name: str


# ============================================
# Model & Embeddings Setup
# ============================================

embeddings = HuggingFaceEmbeddings(model_name="Alibaba-NLP/gte-modernbert-base")

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"), 
    os.getenv("SUPABASE_SERVICE_KEY")
)

vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="gaia_documents",
    query_name="match_documents_langchain",
)

# Updated to use Qwen 3 32B Instruct as requested
model_id = "Qwen/Qwen3-32B-Instruct"
llm = HuggingFaceEndpoint(
    repo_id=model_id,
    temperature=0,
    repetition_penalty=1.03,
    provider="auto",
    huggingfacehub_api_token=os.getenv("HF_INFERENCE_KEY")
)

agent = ChatHuggingFace(llm=llm)

# Bind all tools from the consolidated tools list
agent_with_tools = agent.bind_tools(tools_list)


# ============================================
# Graph Nodes
# ============================================

def retriever_node(state: AgentState) -> AgentState:
    """
    RAG Node: Find similar questions from the vector store.
    
    Searches for similar questions in the Supabase vector store
    and adds the most similar one as context for the agent.
    """
    messages = state.get("messages", [])
    if not messages:
        return {"messages": []}
    
    question_content = messages[0].content if messages else ""
    
    try:
        similar_questions = vector_store.similarity_search(question_content, k=1)
        
        if similar_questions:
            context_message = HumanMessage(
                content=f"Here I provide a similar question and answer for reference:\n\n{similar_questions[0].page_content}"
            )
            return {"messages": [context_message]}
    except Exception as e:
        print(f"Retriever error: {e}")
    
    return {"messages": []}


def processor_node(state: AgentState) -> AgentState:
    """
    Processor Node: Main LLM agent that answers the question.
    
    The agent has access to specific tools for different modalities.
    If a file is attached, information about it is included in the context.
    """
    # Updated path to prompt file in prompts/ directory
    system_prompt = load_prompt("prompts/prompt.yaml")
    messages = state.get("messages", [])
    file_name = state.get("file_name", "")
    task_id = state.get("task_id", "")
    
    # Build complete message list
    full_messages = [system_prompt]
    
    # If a file is attached, inform the agent
    if file_name:
        file_info_message = HumanMessage(
            content=f"Note: A file named '{file_name}' is associated with this question (task_id: {task_id}). "
                    f"You may need to use file processing tools if the question requires analyzing this file."
        )
        full_messages.append(file_info_message)
    
    # Add all conversation messages
    full_messages.extend(messages)
    
    # Invoke LLM with tools
    response = agent_with_tools.invoke(full_messages)
    
    return {"messages": [response]}


# ============================================
# Graph Construction
# ============================================

def agent_graph():
    """
    Build and compile the agent graph.
    
    Flow:
    START -> retriever_node -> processor_node
                                    |
                                    v
                            [tools_condition]
                              /           \
                         (tools)        (END)
                            |
                            v
                      processor_node (loop)
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("retriever_node", retriever_node)
    workflow.add_node("processor_node", processor_node)
    workflow.add_node("tools", ToolNode(tools_list))

    # Add edges
    workflow.add_edge(START, "retriever_node")
    workflow.add_edge("retriever_node", "processor_node")
    
    # Conditional: if LLM wants tools, go to tools node, else END
    workflow.add_conditional_edges(
        "processor_node", 
        tools_condition,
        {"tools": "tools", END: END}
    )
    
    # After tools, return to processor
    workflow.add_edge("tools", "processor_node")

    return workflow.compile()
