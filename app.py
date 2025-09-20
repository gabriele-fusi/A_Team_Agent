"""
Main Streamlit application for A_Team_Agent RAG system.
Provides web UI for document upload, processing, and question-answering.
"""

import streamlit as st
import logging
import time
from pathlib import Path
import traceback

# Import our modules
from src.config import config
from src.document_processor import DocumentProcessor
from src.rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="A_Team_Agent RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline()
    
    if 'document_processor' not in st.session_state:
        st.session_state.document_processor = DocumentProcessor()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    
    if 'vector_store_ready' not in st.session_state:
        # Check if there's already a vector store ready
        st.session_state.vector_store_ready = st.session_state.rag_pipeline.is_ready()
        
        # If not ready but there's an existing vector store, try to initialize
        if not st.session_state.vector_store_ready:
            st.session_state.vector_store_ready = st.session_state.rag_pipeline.initialize_qa_chain()

def display_header():
    """Display the application header."""
    st.title("🤖 A_Team_Agent RAG System")
    st.markdown("""
    A **framework-assisted RAG application** with Streamlit UI, local FAISS vector storage, and OpenAI API integration.
    Upload documents and ask questions to get intelligent answers with source attribution.
    """)

def sidebar_file_upload():
    """Handle file upload in sidebar."""
    st.sidebar.header("📁 Document Upload")
    
    # File uploader
    uploaded_files = st.sidebar.file_uploader(
        "Choose files to upload",
        type=['pdf', 'txt', 'md', 'docx'],
        accept_multiple_files=True,
        help="Upload PDF, TXT, MD, or DOCX files"
    )
    
    if uploaded_files:
        st.sidebar.write(f"Selected {len(uploaded_files)} file(s)")
        
        if st.sidebar.button("Process Documents", type="primary"):
            process_uploaded_files(uploaded_files)

def process_uploaded_files(uploaded_files):
    """Process uploaded files and add to vector store."""
    try:
        with st.spinner("Processing documents..."):
            progress_bar = st.progress(0)
            all_documents = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                # Update progress
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                
                # Get file type
                file_type = Path(uploaded_file.name).suffix.lower()
                
                # Validate file type
                if not st.session_state.document_processor.validate_file_type(uploaded_file.name):
                    st.error(f"Unsupported file type: {file_type}")
                    continue
                
                # Process file
                documents = st.session_state.document_processor.process_uploaded_file(
                    uploaded_file, file_type
                )
                all_documents.extend(documents)
                
                # Add to processed files list
                if uploaded_file.name not in st.session_state.processed_files:
                    st.session_state.processed_files.append(uploaded_file.name)
            
            # Add documents to RAG pipeline
            if all_documents:
                success = st.session_state.rag_pipeline.add_documents(all_documents)
                
                if success:
                    st.session_state.vector_store_ready = True
                    st.success(f"Successfully processed {len(uploaded_files)} file(s) with {len(all_documents)} chunks!")
                    
                    # Display file info
                    file_info = st.session_state.document_processor.get_file_info(all_documents)
                    st.info(f"Total chunks: {file_info['total_chunks']} from {file_info['total_documents']} file(s)")
                    
                    # Refresh the app to show chat interface
                    time.sleep(1)  # Brief pause to show success message
                    st.rerun()
                else:
                    st.error("Failed to add documents to vector store")
            else:
                st.warning("No documents were processed successfully")
                
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        logger.error(f"Document processing error: {traceback.format_exc()}")

def sidebar_vector_store_info():
    """Display vector store information in sidebar."""
    st.sidebar.header("📊 Vector Store Status")
    
    # Get vector store stats
    stats = st.session_state.rag_pipeline.get_vector_store_stats()
    
    if stats['status'] == 'initialized':
        st.sidebar.success("✅ Vector Store Ready")
        st.sidebar.write(f"**Documents:** {stats.get('document_count', 0)}")
        st.sidebar.write(f"**Model:** {stats.get('embedding_model', 'Unknown')}")
    elif stats['status'] == 'not_initialized':
        st.sidebar.warning("⚠️ No Documents Loaded")
        st.sidebar.write("Upload documents to get started")
    else:
        st.sidebar.error("❌ Vector Store Error")
    
    # Processed files
    if st.session_state.processed_files:
        st.sidebar.subheader("📄 Processed Files")
        for filename in st.session_state.processed_files:
            st.sidebar.write(f"• {filename}")
    
    # Clear vector store button
    if st.sidebar.button("Clear All Documents", type="secondary"):
        clear_vector_store()

def clear_vector_store():
    """Clear the vector store and reset state."""
    try:
        success = st.session_state.rag_pipeline.clear_vector_store()
        if success:
            st.session_state.processed_files = []
            st.session_state.vector_store_ready = False
            st.session_state.messages = []
            st.success("Vector store cleared successfully!")
            st.rerun()
        else:
            st.error("Failed to clear vector store")
    except Exception as e:
        st.error(f"Error clearing vector store: {str(e)}")

def sidebar_settings():
    """Display settings in sidebar."""
    st.sidebar.header("⚙️ Settings")
    
    # Model settings
    with st.sidebar.expander("Model Configuration"):
        model_name = st.selectbox(
            "LLM Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
            index=0
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=config.TEMPERATURE,
            step=0.1
        )
        
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=100,
            max_value=2000,
            value=config.MAX_TOKENS,
            step=50
        )
        
        if st.button("Update Settings"):
            st.session_state.rag_pipeline.update_llm_settings(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            st.success("Settings updated!")

def display_chat_interface():
    """Display the main chat interface."""
    st.header("💬 Chat with Your Documents")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Display sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                display_sources(message["sources"])

def display_sources(sources):
    """Display source documents for an answer."""
    if sources:
        with st.expander(f"📚 Sources ({len(sources)} documents)"):
            for i, source in enumerate(sources, 1):
                st.write(f"**Source {i}:**")
                st.write(f"📄 {source['metadata'].get('filename', 'Unknown file')}")
                st.write(f"🔍 Content preview: {source['content']}")
                
                # Additional metadata
                metadata = source['metadata']
                if 'chunk_id' in metadata:
                    st.write(f"📊 Chunk {metadata['chunk_id'] + 1} of {metadata.get('total_chunks', '?')}")
                
                st.divider()

def handle_user_input():
    """Handle user input and generate responses."""
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        
        # Check if vector store is ready
        if not st.session_state.vector_store_ready:
            st.warning("Please upload and process documents before asking questions.")
            return
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Query the RAG pipeline
                    result = st.session_state.rag_pipeline.query_with_context(
                        prompt, 
                        st.session_state.messages[:-1]  # Exclude current message
                    )
                    
                    # Display answer
                    answer = result.get("answer", "No answer generated")
                    st.write(answer)
                    
                    # Display sources
                    sources = result.get("source_documents", [])
                    if sources:
                        display_sources(sources)
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_message = f"Error generating response: {str(e)}"
                    st.error(error_message)
                    logger.error(f"Query error: {traceback.format_exc()}")
                    
                    # Add error message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    })

def display_example_questions():
    """Display example questions when no documents are loaded."""
    if not st.session_state.vector_store_ready:
        st.header("🚀 Getting Started")
        st.markdown("""
        ### How to use A_Team_Agent:
        
        1. **Upload Documents** - Use the sidebar to upload PDF, TXT, MD, or DOCX files
        2. **Process Documents** - Click "Process Documents" to create embeddings
        3. **Ask Questions** - Use the chat interface to query your documents
        
        ### Example Questions (after uploading documents):
        - "What is the main topic of this document?"
        - "Can you summarize the key points?"
        - "What does the document say about [specific topic]?"
        - "Are there any recommendations mentioned?"
        """)

def main():
    """Main application function."""
    try:
        # Validate configuration
        config.validate_config()
        
        # Initialize session state
        initialize_session_state()
        
        # Display header
        display_header()
        
        # Sidebar
        with st.sidebar:
            sidebar_file_upload()
            sidebar_vector_store_info()
            sidebar_settings()
        
        # Main content
        if st.session_state.vector_store_ready:
            display_chat_interface()
            handle_user_input()
        else:
            display_example_questions()
        
        # Footer
        st.markdown("---")
        st.markdown("Built with ❤️ by the A-Team using Streamlit, LangChain, FAISS, and OpenAI")
        
    except ValueError as e:
        st.error(f"Configuration Error: {str(e)}")
        st.info("Please check your .env file and ensure all required environment variables are set.")
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        logger.error(f"Application error: {traceback.format_exc()}")

if __name__ == "__main__":
    main()