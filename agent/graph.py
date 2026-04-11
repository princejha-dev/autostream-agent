from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage
from .state import AgentState
from .intent import detect_intent
from .rag import rag_node
from .tools import collect_info_node, tool_node

def decision_node(state: AgentState) -> str:
    if state.get("is_high_intent") or state.get("intent") == "high_intent":
        return "check_user_data"
            
    intent = state.get("intent")
    if intent == "product_query":
        return "rag_node"
    elif intent == "greeting":
        return "greeting_node"
    else:
        return "greeting_node"

def check_user_data_node(state: AgentState) -> dict:
    """A pass-through node for structural clarity as per the expected diagram."""
    return {}

def check_user_data_route(state: AgentState) -> str:
    user_data = state.get("user_data", {})
    if user_data.get("name") and user_data.get("email") and user_data.get("platform"):
        return "tool_node"
    return "collect_info"

def greeting_node(state: AgentState) -> dict:
    return {"messages": [AIMessage(content="Hello! I am here to help you find the right plan or answer any questions you have. How can I assist you today?")]}

def compile_graph():
    builder = StateGraph(AgentState)
    
    # Nodes
    builder.add_node("detect_intent", detect_intent)
    builder.add_node("check_user_data", check_user_data_node)
    builder.add_node("rag_node", rag_node)
    builder.add_node("collect_info", collect_info_node)
    builder.add_node("tool_node", tool_node)
    builder.add_node("greeting_node", greeting_node)
    
    # Edges
    builder.add_edge(START, "detect_intent")
    
    # decision_node acts as the conditional router extending from detect_intent
    builder.add_conditional_edges(
        "detect_intent", 
        decision_node, 
        {
            "rag_node": "rag_node",
            "check_user_data": "check_user_data",
            "greeting_node": "greeting_node"
        }
    )
    
    # check_user_data conditionally routes to collect_info or tool_node
    builder.add_conditional_edges(
        "check_user_data",
        check_user_data_route,
        {
            "collect_info": "collect_info",
            "tool_node": "tool_node"
        }
    )
    
    # All terminal nodes go to END according to the specified workflow
    builder.add_edge("collect_info", END)
    builder.add_edge("rag_node", END)
    builder.add_edge("tool_node", END)
    builder.add_edge("greeting_node", END)
    
    return builder.compile()
