import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from agent.graph import compile_graph

# Load env
load_dotenv()

st.set_page_config(
    page_title="AutoStream - Contact Sales",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #555;
            margin-bottom: 2rem;
        }
        .status-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9rem;
        }
        .status-active {
            background-color: #d1f2eb;
            color: #0f5132;
        }
        .lead-success-box {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            padding: 2rem;
            border-radius: 1rem;
            margin: 1.5rem 0;
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
            border: none;
        }
        .lead-success-box h2 {
            margin-top: 0;
            font-size: 1.8rem;
        }
        .lead-info-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 1rem;
            margin-top: 1.5rem;
        }
        .lead-info-item {
            background-color: rgba(255, 255, 255, 0.1);
            padding: 1rem;
            border-radius: 0.75rem;
            border-left: 3px solid white;
        }
        .lead-info-label {
            font-size: 0.9rem;
            opacity: 0.9;
            margin-bottom: 0.5rem;
        }
        .lead-info-value {
            font-size: 1.1rem;
            font-weight: bold;
        }
        .chat-message {
            display: flex;
            margin: 1rem 0;
        }
        .chat-message.user {
            justify-content: flex-end;
        }
        .chat-message.assistant {
            justify-content: flex-start;
        }
        .message-content {
            padding: 1rem;
            border-radius: 0.75rem;
            max-width: 70%;
            word-wrap: break-word;
        }
        .user-message {
            background-color: #1f77b4;
            color: white;
        }
        .assistant-message {
            background-color: #f0f0f0;
            color: black;
        }
        @media (max-width: 768px) {
            .lead-info-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Title
col1, col2 = st.columns([4, 1])
with col1:
    st.markdown('<div class="main-header">🤖 AutoStream AI Sales Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Your AI-powered product expert - Always here to help</div>', unsafe_allow_html=True)

# Sidebar - Customer focused
with st.sidebar:
    st.markdown('<h3 style="color: #1f77b4; margin-top: 0;">Welcome!</h3>', unsafe_allow_html=True)
    st.write("Hi! 👋 I'm here to answer your questions about our products, pricing, and help you get started.")
    
    st.divider()
    
    st.markdown('<h3 style="color: #1f77b4;">📋 Quick Info</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Status", "🟢 Online", help="Always available to chat")
    with col2:
        st.metric("Response", "Instant", help="Real-time answers")
    
    st.divider()
    
    st.markdown('<h3 style="color: #1f77b4;">💬 How can I help?</h3>', unsafe_allow_html=True)
    st.markdown("""
    - 📦 Ask about our plans
    - 💰 Pricing information
    - ❓ General questions
    - 📝 Get started today
    """)
    
    st.divider()
    
    if st.button("🔄 Start New Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.state = {
            "messages": [],
            "intent": None,
            "user_data": {"name": None, "email": None, "platform": None},
            "is_high_intent": False,
            "lead_collected": False
        }
        st.rerun()

# Initialize graph (only once)
@st.cache_resource
def load_agent():
    return compile_graph()

app = load_agent()

# Initialize session state
if "state" not in st.session_state:
    st.session_state.state = {
        "messages": [],
        "intent": None,
        "user_data": {"name": None, "email": None, "platform": None},
        "is_high_intent": False,
        "lead_collected": False
    }

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display lead collection status BEFORE chat
if st.session_state.state.get("lead_collected"):
    user_data = st.session_state.state.get("user_data", {})
    st.markdown("""
    <div class="lead-success-box">
        <h2>✅ Thank You!</h2>
        <p>We've received your information and will be in touch shortly. Our team will reach out to discuss the best plan for your needs.</p>
        <div class="lead-info-grid">
    """, unsafe_allow_html=True)
    
    if user_data.get("name"):
        st.markdown(f"""
            <div class="lead-info-item">
                <div class="lead-info-label">Name</div>
                <div class="lead-info-value">{user_data['name']}</div>
            </div>
        """, unsafe_allow_html=True)
    
    if user_data.get("email"):
        st.markdown(f"""
            <div class="lead-info-item">
                <div class="lead-info-label">Email</div>
                <div class="lead-info-value">{user_data['email']}</div>
            </div>
        """, unsafe_allow_html=True)
    
    if user_data.get("platform"):
        st.markdown(f"""
            <div class="lead-info-item">
                <div class="lead-info-label">Platform</div>
                <div class="lead-info-value">{user_data['platform']}</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    st.divider()

# Display chat history
st.markdown("### 💬 Conversation")
chat_container = st.container()
with chat_container:
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.chat_message("user").markdown(chat['content'])
        else:
            st.chat_message("assistant").markdown(chat['content'])

# Chat input
user_input = st.chat_input("Type your message here...", key="chat_input")

if user_input:
    # Show user message
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Add to state
    st.session_state.state["messages"].append(HumanMessage(content=user_input))

    with st.spinner("🤖 Agent is thinking..."):
        try:
            # Invoke LangGraph agent
            result = app.invoke(st.session_state.state)

            # Update state
            st.session_state.state = result

            # Get last AI message
            messages = result.get("messages", [])
            ai_reply = ""

            if messages and messages[-1].type == "ai":
                ai_reply = messages[-1].content

            # Display AI response
            st.chat_message("assistant").markdown(ai_reply)
            st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})

            st.rerun()

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Please check that all required API keys are set in your .env file (GROQ_API_KEY, GEMINI_API_KEY)")
