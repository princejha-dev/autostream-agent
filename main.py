import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from agent.graph import compile_graph

import warnings
warnings.filterwarnings("ignore")

# Load environment variables (e.g. OPENAI_API_KEY)
load_dotenv()

def print_messages(messages):
    """Utility to print the last AIMessage"""
    if not messages: return
    last_msg = messages[-1]
    # In some execution paths, we might have multiple messages. We just take the latest.
    if last_msg.type == "ai":
        print(f"\nAgent: {last_msg.content}")

def main():
    print("Welcome to Autostream-agent! Type 'quit' to exit.")
    
    # Needs GROQ_API_KEY
    if not os.getenv("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY not found in environment! Groq features will fail.")
        
    app = compile_graph()
    
    state = {
        "messages": [],
        "intent": None,
        "user_data": {"name": None, "email": None, "platform": None},
        "is_high_intent": False,
        "lead_collected": False
    }

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                break
                
            # Append user message
            state["messages"].append(HumanMessage(content=user_input))
            
            # Invoke Graph
            result = app.invoke(state)
            
            # Update state with the result for the next iteration
            state = result
            
            print_messages(state.get("messages", []))
            
            if state.get("lead_collected"):
                print("\n[Application] Lead collection flow concluded.")
                break
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {e}")
            break

if __name__ == "__main__":
    main()
