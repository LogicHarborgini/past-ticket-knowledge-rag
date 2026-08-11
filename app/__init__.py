"""
Past Ticket Knowledge (PTK) — a RAG pipeline over historical support tickets.

Module layout, in the order data flows through it:

    corpus.py       the synthetic historical ticket set
    embeddings.py   embedding model selection (Bedrock / Ollama / keyword)
    vectorstore.py  Chroma collection + retriever
    ingest.py       embeds the corpus into the collection (run once)
    ptk_chain.py    the LCEL RAG chain and its traced entry point
    models.py       Pydantic request/response contracts
    main.py         FastAPI service
"""
