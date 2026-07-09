# past-ticket-knowledge-rag

> RAG pipeline that retrieves semantically similar historical support tickets 
> and summarises past resolutions using Amazon Bedrock.

## Problem Statement

Support engineers need to find how similar issues were resolved in the past, 
but searching thousands of historical tickets manually is slow and inconsistent.
PTA automates this using semantic similarity search.

## Architecture


                        PTA — Past Ticket Analysis
                        RAG Pipeline

  Support Engineer                              Historical Ticket Corpus
       │                                               │
       │  Query: "database timeout errors"            │ (Indexed offline)
       ▼                                               ▼
 ┌──────────┐    ┌─────────────────┐    ┌─────────────────────────┐
 │ FastAPI  │───▶│ Amazon Titan    │    │  Amazon OpenSearch       │
 │ POST /   │    │ Embeddings      │    │  (k-NN Vector Index)     │
 │ /search  │    │                 │    │                          │
 └──────────┘    │ Query → Vector  │───▶│  Cosine Similarity       │
                 │ [0.12, 0.87...] │    │  Top-5 Similar Tickets   │
                 └─────────────────┘    └────────────┬────────────┘
                                                     │
                                        Similar Tickets Retrieved
                                                     │
                                                     ▼
                                        ┌────────────────────────┐
                                        │  Amazon Bedrock        │
                                        │  Claude 3 Sonnet       │
                                        │                        │
                                        │  Prompt:               │
                                        │  "Given these similar  │
                                        │   past tickets and     │
                                        │   their resolutions,   │
                                        │   summarise how to     │
                                        │   resolve: [query]"    │
                                        └────────────┬───────────┘
                                                     │
                                                     ▼
                                        Summarised Past Resolutions
                                        Delivered to Engineer

Flow:
  1. Engineer submits a natural-language query
  2. Amazon Titan Embeddings converts query to a dense vector
  3. OpenSearch k-NN search finds 5 most semantically similar historical tickets
  4. Similar tickets + original query sent to Claude 3 Sonnet on Bedrock
  5. Bedrock summarises the relevant past resolutions
  6. Summary returned to engineer

Key Design Decisions:
  - RAG (Retrieval-Augmented Generation): grounds LLM response in real past tickets
  - Titan Embeddings: AWS-native embedding model, no external dependencies
  - OpenSearch k-NN: vector search within existing AWS infrastructure
  - Cosine similarity: measures semantic closeness, not keyword overlap

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