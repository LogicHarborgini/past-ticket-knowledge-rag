# past-ticket-knowledge-rag

> RAG pipeline that retrieves semantically similar historical support tickets 
> and summarises past resolutions using Amazon Bedrock.

## Problem Statement

Support engineers need to find how similar issues were resolved in the past, 
but searching thousands of historical tickets manually is slow and inconsistent.
PTA automates this using semantic similarity search.

## Architecture

*(Diagram coming Day 3)*

```
[User Query] → [Titan Embeddings] → [OpenSearch k-NN] → [Similar Tickets] → [Bedrock Summary]
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Embedding Model | Amazon Titan Embeddings |
| Vector Store | Amazon OpenSearch |
| LLM (Summarisation) | Amazon Bedrock (Claude) |
| Retrieval Pattern | RAG (Retrieval-Augmented Generation) |
| Language | Python 3.11 |

## Project Status

🔨 In active development — Week 1

## Note

This is an open-source portfolio version. The production system runs at Cleo on proprietary ticket data.
```