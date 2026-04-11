from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class UserData(TypedDict):
    name: str | None
    email: str | None
    platform: str | None

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    intent: str | None
    user_data: UserData
    is_high_intent: bool
    lead_collected: bool
