from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
import os
from .state import AgentState

class IntentClassification(BaseModel):
    intent: str = Field(description="The classified intent: 'greeting', 'product_query', or 'high_intent'.")
    is_high_intent: bool = Field(description="True if the user intends to buy, sign up, start a trial, or purchase a plan.")

def detect_intent(state: AgentState) -> dict:
    """
    Node to detect the intent of the user.
    Uses ChatOpenAI to process the latest messages and classify the intent.
    """
    messages = state.get("messages", [])
    
    # We only need the conversation context and the last user message to determine intent
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    structured_llm = llm.with_structured_output(IntentClassification)
    
    system_prompt = (
        "You are an intent classification system for an AI assistant. "
        "Classify the user's latest message into one of three categories:\n"
        "1. 'greeting': The user is saying hi, hello, or asking how you are.\n"
        "2. 'product_query': The user is asking about pricing, plans, features, or policies.\n"
        "3. 'high_intent': The user explicitly wants to buy, purchase, sign up, or get a specific plan.\n\n"
        "If a previous turn established high intent, and the user is just answering your questions "
        "(providing their email, name, platform), maintain 'high_intent'.\n"
    )
    
    classification = structured_llm.invoke([SystemMessage(content=system_prompt)] + messages)
    
    # If it's already high intent, keep it.
    if state.get("is_high_intent"):
        return {"intent": "high_intent", "is_high_intent": True}
        
    return {
        "intent": classification.intent,
        "is_high_intent": classification.is_high_intent
    }
