import os
import pandas as pd
import streamlit as st
import logging
import tempfile
import uuid
from datetime import datetime
from typing import List, Dict

# Import the RAGChatbot class from rag_chatbot.py
from rag_chatbot import RAGChatbot

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Create session state variables if they don't exist
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'conversations' not in st.session_state:
    st.session_state.conversations = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'temp_file_path' not in st.session_state:
    st.session_state.temp_file_path = None

def save_conversations():
    """Save the current conversations to a file"""
    if not st.session_state.conversations:
        st.warning("No conversations to save.")
        return
    
    try:
        # Create a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversations_{timestamp}.txt"
        
        # Use the chatbot to save conversations
        if st.session_state.chatbot:
            st.session_state.chatbot.save_conversation(
                st.session_state.conversations, 
                filename
            )
            st.success(f"Conversations saved successfully to {filename}")
        else:
            st.error("Chatbot not initialized. Cannot save conversations.")
    except Exception as e:
        st.error(f"Error saving conversations: {str(e)}")

def handle_user_input():
    """Process user input and generate a response"""
    user_question = st.session_state.user_input
    
    if user_question:
        # Add user question to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        # Clear the input field
        st.session_state.user_input = ""
        
        # Display a spinner while processing
        with st.spinner("Thinking..."):
            try:
                # Get response from chatbot
                if st.session_state.chatbot:
                    bot_response = st.session_state.chatbot.query(user_question)
                    
                    # Add bot response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": bot_response})
                    
                    # Save this conversation
                    st.session_state.conversations.append({
                        "question": user_question,
                        "answer": bot_response
                    })
                else:
                    st.error("Please upload a CSV file to initialize the chatbot first.")
            except Exception as e:
                st.error(f"Error processing your question: {str(e)}")

def handle_file_upload():
    """Process the uploaded CSV file and initialize the chatbot"""
    uploaded_file = st.session_state.uploaded_file
    
    if uploaded_file:
        try:
            # Display a spinner while processing
            with st.spinner("Processing your file..."):
                # Save the uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_file_path = tmp_file.name
                
                # Store the temp file path in session state
                st.session_state.temp_file_path = temp_file_path
                
                # Get API key from secrets or environment
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    api_key = st.secrets.get("GOOGLE_API_KEY", "")
                
                if not api_key:
                    st.error("Google API key not found. Please set it in your environment or Streamlit secrets.")
                    return
                
                # Initialize the chatbot with the temporary file
                st.session_state.chatbot = RAGChatbot(
                    google_api_key=api_key,
                    data_path=temp_file_path
                )
                
                # Reset conversations
                st.session_state.conversations = []
                
                # Initialize messages if not already done
                if 'messages' not in st.session_state:
                    st.session_state.messages = []
                
                st.success("Chatbot initialized successfully! You can now ask questions about your data.")
        except Exception as e:
            st.error(f"Error initializing chatbot: {str(e)}")

def main():
    """Main function to run the Streamlit app"""
    
    # Title and description
    st.title("📚 RAG Chatbot with Gemini")
    st.markdown("""
    Upload your CSV file to create a knowledge base, then chat with the AI to get answers based on your data.
    
    This application uses Retrieval-Augmented Generation (RAG) to provide more accurate and contextual responses.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # File uploader
        st.file_uploader(
            "Upload your CSV knowledge base",
            type=["csv"],
            key="uploaded_file",
            on_change=handle_file_upload,
            help="Upload a CSV file containing your knowledge base"
        )
        
        # Display session info
        st.subheader("Session Information")
        st.info(f"Session ID: {st.session_state.session_id[:8]}...")
        
        # End session button
        if st.button("End Session & Save Conversations"):
            save_conversations()
            
            # Clean up temporary files
            if st.session_state.temp_file_path and os.path.exists(st.session_state.temp_file_path):
                try:
                    os.remove(st.session_state.temp_file_path)
                    logger.info(f"Removed temporary file: {st.session_state.temp_file_path}")
                except Exception as e:
                    logger.error(f"Error removing temporary file: {str(e)}")
            
            # Reset session
            st.session_state.chatbot = None
            st.session_state.conversations = []
            st.session_state.messages = []
            st.session_state.temp_file_path = None
            st.session_state.session_id = str(uuid.uuid4())
            
            st.success("Session ended and reset successfully!")
            st.rerun()
    
    # Main content area - Chat interface
    chat_container = st.container()
    
    # Initialize messages list if not exists
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if st.session_state.chatbot:
        user_input = st.chat_input(
            "Ask a question about your data...",
            key="user_input",
            on_submit=handle_user_input
        )
    else:
        st.info("Please upload a CSV file to start chatting.")
    
    # Display dataset preview if chatbot is initialized
    if st.session_state.chatbot and st.session_state.temp_file_path:
        with st.expander("Preview Dataset"):
            try:
                df = pd.read_csv(st.session_state.temp_file_path)
                st.dataframe(df.head(10))
                st.text(f"Total rows: {len(df)}, Columns: {', '.join(df.columns)}")
            except Exception as e:
                st.error(f"Error previewing dataset: {str(e)}")

if __name__ == "__main__":
    main()