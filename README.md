# Streamlit RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, LangChain, and Google's Gemini API. This application allows users to upload CSV files to create a knowledge base and then interact with the data through natural language queries.

## Features

- 📁 CSV file upload for knowledge base creation
- 💬 Interactive chat interface
- 🔍 RAG-based question answering
- 💾 Session conversation saving
- 📊 Data preview functionality

## Project Structure

```
rag-chatbot-streamlit/
├── app.py                     # Main Streamlit application
├── rag_chatbot.py             # RAG Chatbot implementation
├── requirements.txt           # Project dependencies
├── .streamlit/
│   ├── config.toml            # Streamlit configuration
│   └── secrets.toml.example   # Example secrets file (rename to secrets.toml)
└── .env.example               # Example environment variables (rename to .env)
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/rag-chatbot-streamlit.git
   cd rag-chatbot-streamlit
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up your Google API key:
   - Option 1: Rename `.env.example` to `.env` and add your API key
   - Option 2: Rename `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add your API key

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501).

3. Upload a CSV file containing your knowledge base data.

4. Start chatting with the bot by asking questions related to your data.

5. When you're done, click "End Session & Save Conversations" to save the chat history.

## Requirements for CSV Files

- The CSV should have clear column headers.
- Each row should represent a coherent piece of information.
- The data should be clean and well-structured for best results.

## Deployment

You can deploy this application on Streamlit Cloud or any other platform that supports Streamlit applications:

1. Push your code to GitHub.
2. Connect your repository to Streamlit Cloud.
3. Configure your secrets in the Streamlit Cloud dashboard.
4. Deploy the application.

## Acknowledgments

- This project uses [LangChain](https://github.com/langchain-ai/langchain) for RAG implementation.
- Powered by Google's [Gemini API](https://ai.google.dev/gemini-api) for natural language understanding and generation.
- Built with [Streamlit](https://streamlit.io/) for the web interface.
