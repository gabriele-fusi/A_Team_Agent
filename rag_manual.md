# RAG System Manual

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that combines:
1. **Information Retrieval** - Finding relevant documents
2. **Language Generation** - Creating natural language responses

## How This System Works

1. **Document Upload** - Users upload files through the web interface
2. **Processing** - Documents are chunked and embedded using OpenAI
3. **Storage** - Embeddings are stored in a local FAISS vector database
4. **Query** - User questions are converted to embeddings
5. **Retrieval** - Similar document chunks are found
6. **Generation** - OpenAI generates answers based on retrieved context

## Benefits

- No internet dependency for vector storage (local FAISS)
- Fast retrieval and response times
- Source attribution for transparency
- Support for conversation history
- Easy document management through web UI

## Usage Tips

- Upload multiple related documents for better context
- Ask specific questions for more accurate answers
- Check the source attributions to verify information
- Use the conversation history for follow-up questions