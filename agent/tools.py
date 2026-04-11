from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, AIMessage
from pydantic import BaseModel, Field
from .state import AgentState

class UserInfoExtraction(BaseModel):
    name: str | None = Field(default=None, description="The user's name if mentioned.")
    email: str | None = Field(default=None, description="The user's email if mentioned.")
    platform: str | None = Field(default=None, description="The user's platform (e.g. YouTube, Twitter, Instagram) if mentioned.")

def mock_lead_capture(name: str, email: str, platform: str):
    print(f"\n[API CALL] Lead captured successfully: {name}, {email}, {platform}\n")
    return True

def collect_info_node(state: AgentState) -> dict:
    """Extracts user info and asks for missing data."""
    messages = state.get("messages", [])
    user_data = state.get("user_data", {})
    
    if not user_data:
        user_data = {"name": None, "email": None, "platform": None}
    
    # Extract data from the conversation
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    extractor = llm.with_structured_output(UserInfoExtraction)
    
    system_prompt = (
        "Extract the user's name, email, and platform from the conversation if explicitly provided. "
        "Return null for any field that is not found."
    )
    
    extracted = extractor.invoke([SystemMessage(content=system_prompt)] + messages)
    
    if extracted.name and not user_data.get("name"): 
        user_data["name"] = extracted.name
    if extracted.email and not user_data.get("email"): 
        user_data["email"] = extracted.email
    if extracted.platform and not user_data.get("platform"): 
        user_data["platform"] = extracted.platform
    
    missing_fields = []
    if not user_data.get("name"): missing_fields.append("name")
    if not user_data.get("email"): missing_fields.append("email")
    if not user_data.get("platform"): missing_fields.append("platform")
    
    if missing_fields:
        # Prompt for the first missing field
        next_missing = missing_fields[0]
        if next_missing == "name":
            msg = "Great! To get you started, could I please have your name?"
        elif next_missing == "email":
            name_ref = user_data.get("name", "there")
            msg = f"Thanks {name_ref}! What is your email address?"
        elif next_missing == "platform":
            msg = "Got it! Which platform are you using (e.g., YouTube, Instagram, etc.)?"
            
        return {
            "user_data": user_data,
            "messages": [AIMessage(content=msg)]
        }
    else:
        # We don't append an AIMessage here because the tool_node will handle the final message.
        return {
            "user_data": user_data
        }

def tool_node(state: AgentState) -> dict:
    user_data = state.get("user_data", {})
    name = user_data.get("name")
    email = user_data.get("email")
    platform = user_data.get("platform")
    
    mock_lead_capture(name, email, platform)
    
    msg = f"You're all set, {name}! We've captured your details for {platform} and will follow up with you at {email}."
    return {
        "lead_collected": True,
        "messages": [AIMessage(content=msg)]
    }
