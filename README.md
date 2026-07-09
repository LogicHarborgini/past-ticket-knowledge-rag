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

## Problem Statement

When a support engineer faces an unfamiliar issue, finding how similar problems 
were resolved in the past requires manual searching through thousands of historical 
tickets — slow, inconsistent, and error-prone.

**Past Ticket Knowledge RAG** solves this using a RAG (Retrieval-Augmented Generation) 
pipeline: the engineer's query is converted to a vector using Amazon Titan Embeddings, 
semantically similar historical tickets are retrieved from Amazon OpenSearch, and 
Amazon Bedrock summarises the relevant past resolutions.