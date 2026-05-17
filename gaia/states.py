from typing import Annotated, TypedDict, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document

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
        file_path: Local filesystem path to the downloaded file (empty if no file or download failed)
        retrieved_docs: List of candidate documents from the retriever node
        final_answer: GAIA-formatted answer produced by the formatter node
    """
    messages: Annotated[list[BaseMessage], add_messages]
    task_id: str
    file_name: str
    file_path: str
    retrieved_docs: List[Document]
    final_answer: str