# COURSE : Generative AI for the Enterprise
# MIDTERM EXAM : Multi-Agent RAG System
# NAME : Tanish Jagadheshan
# DATE : 01-30-2026

# MULTI-AGENT RESEARCH ASSISTANT
# Using LangGraph

"""
KEY ARCHITECTURAL CHOICES:

1.  Local Stack: 
    - LLM: 'ChatOllama' running Llama 3.2. Chosen for zero-cost operation and low latency on local hardware.
    - Embeddings: 'HuggingFaceEmbeddings' (all-MiniLM-L6-v2). Runs locally ensuring no data leaves the system.
    
2.  LangGraph:
    - Used 'StateGraph' instead of linear sequential chains. This allows for conditional logic 
      (Routing) and shared state (AgentState) passing between nodes.

      # Framework Choice: LangGraph was chosen over CrewAI to implement a deterministic system with conditional edges, 
      allowing precise control over the routing logic compared to CrewAI's autonomous delegation.

3.  Multi Agent System:
    - Instead of one massive prompt, the task is broken into:
        * Router: Intent classification
        * Retriever: Fetches raw data
        * Analyzer: Validates data relevance to reduces hallucinations
        * Synthesizer: Drafts the final response
"""

# Import Libraries
import streamlit as st
import os
import tempfile
from typing import List, TypedDict, Literal

# LangChain and LangGraph Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings



# 1. CONFIGURATION & SETUP
st.set_page_config(page_title="Multi-Agent Research Assistant", layout="wide")

# Initialize Local Models
try:
    # i) LLM - Using Ollama locally (llama3.2)
    llm = ChatOllama(model="llama3.2", temperature=0)
    
    # ii) Embeddings - Using HuggingFace (Local)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
except Exception as e:
    st.error(f"Failed to initialize local models. Error: {e}")
    st.stop()



# 2. VECTOR DATABASE LAYER SETUP (CHROMA DB)
def process_documents(uploaded_files):
    """
    Handles PDF ingestion, chunking, and vector storage locally.
    """
    documents = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            try:
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
                return None

    if not documents:
        st.error("No valid text found in uploaded documents.")
        return None

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    splits = text_splitter.split_documents(documents)

    # Store in Chroma Vector DB locally
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="midterm_local_collection"
    )
    
    return vectorstore.as_retriever(search_kwargs={"k": 5})



# 3. MULTI-AGENT SYSTEM IMPLEMENTATION
class AgentState(TypedDict):
    question: str
    documents: List[str]
    analysis: str
    generation: str
    decision: str
    agent_status: str
    relevant_docs_found: bool


# NODE 0 : UNIVERSAL ROUTER
def router_node(state: AgentState):
    print("---ROUTER---")
    question = state["question"].lower()
    
    # i) GENERIC KEYWORD CHECK (Explicit intent)
    generic_triggers = [
        "pdf", "file", "document", "paper", "report", "article", 
        "context", "provided", "upload", "this"
    ]
    
    if any(trigger in question for trigger in generic_triggers):
        decision = "vectorstore"
        reason = "User explicitly asked for file context"
    
    else:
        # ii) SEMANTIC AI CHECK
        prompt = ChatPromptTemplate.from_template(
            """You are a routing agent.
            User Question: {question}
            
            Is this question:
            A) GENERAL KNOWLEDGE (Cooking, history, coding, math, greeting) -> 'general_chat'
            B) SPECIFIC CONTEXT (Asking about 'the study', 'the results', 'the project') -> 'vectorstore'
            
            Return ONLY one word: 'vectorstore' or 'general_chat'."""
        )
        
        try:
            chain = prompt | llm
            response = chain.invoke({"question": question})
            decision = response.content.strip().lower()
            
            if "general" in decision:
                decision = "general_chat"
            else:
                decision = "vectorstore"
            reason = "AI Semantics"
        except:
            decision = "vectorstore"
            reason = "Fallback"

    return {
        "decision": decision,
        "agent_status": f"**Router Agent:** Routing to '{decision}' ; Reason: {reason}"
    }


# NODE 1 : RETRIEVER
def retrieve_node(state: AgentState, retriever):
    question = state["question"]
    try:
        docs = retriever.invoke(question)
        doc_texts = [d.page_content for d in docs]
        return {
            "documents": doc_texts,
            "agent_status": "**Retriever Agent:** Found relevant document chunks"
        }
    except Exception as e:
        return {"documents": [], "agent_status": f"Error: {e}"}


# NODE 2 : ANALYZER
def analyze_node(state: AgentState):
    question = state["question"]
    documents = state["documents"]
    
    # If no docs found, we immediately flag as irrelevant
    if not documents:
        return {
            "analysis": "No documents found.",
            "relevant_docs_found": False,
            "agent_status": "**Analyzer Agent:** No content found. Switching to General Knowledge."
        }

    context = "\n\n".join(documents)
    
    # Ask LLM: "Is this context actually useful for this question?"
    prompt = ChatPromptTemplate.from_template(
        """User Question: {question}
        Context: {context}
        
        Task: Determine if the context contains the answer.
        1. If YES: Summarize the answer from the text.
        2. If NO (Context is unrelated): Say "IRRELEVANT".
        """
    )
    chain = prompt | llm
    result = chain.invoke({"question": question, "context": context})
    analysis = result.content
    
    # Logic: If the analysis says "irrelevant", we switch tracks
    is_irrelevant = "irrelevant" in analysis.lower()
    
    if is_irrelevant:
         return {
            "analysis": analysis,
            "relevant_docs_found": False,
            "agent_status": "**Analyzer Agent:** Docs are irrelevant. Switching to General Knowledge."
        }
    else:
        return {
            "analysis": analysis,
            "relevant_docs_found": True,
            "agent_status": "**Analyzer Agent:** Relevance Confirmed. Proceeding to synthesis."
        }


# NODE 3 : SYNTHESIZER
def synthesize_node(state: AgentState):
    question = state["question"]
    analysis = state["analysis"]
    
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based ONLY on the notes below.
        Question: {question}
        Notes: {analysis}"""
    )
    chain = prompt | llm
    result = chain.invoke({"question": question, "analysis": analysis})
    return {
        "generation": result.content,
        "agent_status": "**Synthesizer Agent:** Drafting final response."
    }


# NODE 4: GENERALIST
def generalist_node(state: AgentState):
    question = state["question"]
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. 
        User Question: {question}
        Answer the question using your general knowledge."""
    )
    chain = prompt | llm
    result = chain.invoke({"question": question})
    return {
        "generation": result.content,
        "agent_status": "**Generalist Agent:** Answering from general knowledge."
    }


# GRAPH CONSTRUCTION
def build_graph(retriever_instance):
    workflow = StateGraph(AgentState)
    workflow.add_node("router", router_node)
    workflow.add_node("retriever", lambda state: retrieve_node(state, retriever_instance))
    workflow.add_node("analyzer", analyze_node)
    workflow.add_node("synthesizer", synthesize_node)
    workflow.add_node("generalist", generalist_node)

    workflow.set_entry_point("router")
    

    # Edge 1: Router Decision
    def decide_route(state):
        return "vectorstore" if state["decision"] == "vectorstore" else "general_chat"

    workflow.add_conditional_edges(
        "router",
        decide_route,
        {"vectorstore": "retriever", "general_chat": "generalist"}
    )
    
    workflow.add_edge("retriever", "analyzer")
    

    # Edge 2: Analyzer Self Correction Decision
    # If Analyzer says docs are irrelevant, we jump to Generalist
    def check_relevance(state):
        if state["relevant_docs_found"]:
            return "synthesizer"
        else:
            return "generalist"

    workflow.add_conditional_edges(
        "analyzer",
        check_relevance,
        {"synthesizer": "synthesizer", "generalist": "generalist"}
    )

    workflow.add_edge("synthesizer", END)
    workflow.add_edge("generalist", END)
    return workflow.compile()



# 4. STREAMLIT INTERFACE

st.title("Multi-Agent Research Assistant")
st.markdown("Analyze PDFs and get answers!")

with st.sidebar:
    st.header("📂 Knowledge Base")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files and st.button("Index Documents"):
        with st.status("Processing Documents", expanded=True) as status:
            st.write("Reading PDF files")
            retriever = process_documents(uploaded_files)
            
            if retriever:
                st.write("Splitting text into chunks")
                st.write("Embedding data into Vector Store")
                st.session_state['retriever'] = retriever
                status.update(label="Indexing Complete", state="complete", expanded=False)
            else:
                status.update(label="Indexing Failed", state="error")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if 'retriever' in st.session_state:
        app = build_graph(st.session_state['retriever'])
        status_container = st.status("Local Agents Working", expanded=True)
        
        final_response = None
        source_docs = [] 
        show_sources = False
        
        try:
            for event in app.stream({"question": prompt}):
                for node_name, node_output in event.items():
                    if "agent_status" in node_output:
                        status_container.write(node_output["agent_status"])
                    
                    if "generation" in node_output:
                        final_response = node_output["generation"]
                    
                    if node_name == "retriever" and "documents" in node_output:
                        source_docs = node_output["documents"]
                    
                    if node_name == "analyzer" and "relevant_docs_found" in node_output:
                        show_sources = node_output["relevant_docs_found"]
                        
            status_container.update(label="Response Generated", state="complete", expanded=False)
            
            if final_response:
                st.session_state.messages.append({"role": "assistant", "content": final_response})
                with st.chat_message("assistant"):
                    st.markdown(final_response)
                    
                    # Only show sources if Analyzer said "Yes, these are relevant"
                    if show_sources and source_docs:
                        with st.expander("View Source Document Chunks"):
                            for i, doc in enumerate(source_docs):
                                st.markdown(f"**Chunk {i+1}**")
                                st.caption(doc[:600] + "...")
                                st.divider()
                                

        ##Error Handling incase of edge cases (incorrect uploads, no upload.)                        
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please upload a PDF first.")