# RAG-based Chatbot with LangChain and Gemini LLM

This repository contains the implementation of a Retrieval-Augmented Generation (RAG) chatbot using LangChain and Google's Gemini LLM. The chatbot retrieves relevant information from a dataset and uses it to generate accurate and contextually appropriate responses to user queries.

## Project Structure

- `rag_chatbot.py`: Main Python script implementing the RAG chatbot
- `sample_data.csv`: Sample dataset with information about AI, machine learning, and related topics
- `sample_conversations.txt`: Sample output file with example conversations
- `requirements.txt`: List of required Python packages

## Features

- Data loading and preprocessing from CSV files
- Vector embedding creation using Google's Generative AI Embeddings
- Efficient vector storage and retrieval using FAISS
- RAG pipeline implementation with LangChain
- Interactive chat interface for querying the system
- Conversation logging and saving functionality

## Requirements

- Python 3.8+
- Google Generative AI API key
- Required packages (see `requirements.txt`)

## Setup Instructions

1. Clone this repository:
   ```
   git clone https://github.com/Sweekar07/RAGChatSage.git
   cd rag-chatbot
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Google API key:
   - Obtain an API key from the [Google AI Studio](https://ai.google.dev/)
   - Replace `"YOUR_GOOGLE_API_KEY"` in the script with your actual API key

4. Prepare your dataset:
   - Use the provided `sample_data.csv` or replace it with your own dataset
   - Ensure your CSV file is properly formatted with content columns

## Usage

Run the script with:

```
python rag_chatbot.py
```

The script will:
1. Load the dataset
2. Create embeddings and set up the retrieval system
3. Run a few sample queries
4. Enter an interactive mode where you can ask your own questions

To exit the interactive mode, type 'quit', 'exit', or 'q'.

## How It Works

### Data Loading (Task 1)
The system loads data from a CSV file and processes it into a format suitable for embedding and retrieval.

### RAG Setup with LangChain (Task 2)
The implementation uses:
- Google's Generative AI Embeddings to create vector representations of documents
- FAISS for efficient similarity search on the vectors
- LangChain's components to create a retrieval chain

### Chatbot Implementation (Task 3)
The chatbot:
1. Takes a user query
2. Retrieves relevant context from the knowledge base
3. Constructs a prompt with the retrieved context
4. Sends the prompt to the Gemini LLM
5. Returns the generated response to the user

## Example Conversations

Sample questions that can be asked:

- "What is RAG and how does it work?"
- "Explain the LangChain framework"
- "How are vector databases used in AI applications?"
- "What is the difference between embeddings and tokenization?"

## Customization

You can customize the behavior of the chatbot by:
- Modifying the prompt template in the `_setup_chain` method
- Adjusting the number of retrieved documents by changing the `k` parameter
- Creating a more sophisticated document loading process for your specific data format

## References

- [LangChain Documentation](https://python.langchain.com/docs/get_started)
- [Google Generative AI Python SDK](https://ai.google.dev/tutorials/python_quickstart)
- [Google Gemini Embeddings](https://ai.google.dev/gemini-api/docs/embeddings)
- [Langchain Gemini Embeddings](https://python.langchain.com/docs/integrations/text_embedding/google_generative_ai/)
- [ChatGoogleGenerativeAI](https://python.langchain.com/docs/integrations/chat/google_generative_ai/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)