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
        retrieved_docs: List of candidate documents from the retriever node
    """
    messages: Annotated[list[BaseMessage], add_messages]
    task_id: str
    file_name: str
    retrieved_docs: List[Document]