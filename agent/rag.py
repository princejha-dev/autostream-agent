import json
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, SystemMessage
from .state import AgentState

# Global variable to hold the vectorstore to avoid rebuilding every time
vectorstore = None

def get_vectorstore():
    global vectorstore
    if vectorstore is not None:
        return vectorstore
        
    knowledge_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "knowledge.json")
    with open(knowledge_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    docs = []
    # parse the json to create documents
    for category, content in data.items():
        if isinstance(content, dict):
            for k, v in content.items():
                docs.append(Document(page_content=f"{k}: {v}", metadata={"category": category, "topic": k}))
        else:
            docs.append(Document(page_content=content, metadata={"category": category}))
            
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def rag_node(state: AgentState) -> dict:
    """Retrieves relevant context and answers the user's query."""
    messages = state.get("messages", [])
    if not messages:
        return {"messages": []}
        
    latest_query = messages[-1].content
    
    vs = get_vectorstore()
    retrieved_docs = vs.similarity_search(latest_query, k=3)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    system_prompt = (
        "You are a helpful assistant for a company. Use the following context to answer the user's query.\n"
        "If you don't know the answer based on the context, say you don't know.\n\n"
        "Context:\n"
        f"{context}"
    )
    
    msgs = [SystemMessage(content=system_prompt)] + messages
    response = llm.invoke(msgs)
    
    return {"messages": [response]}
