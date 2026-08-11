"""
Application configuration.

Uses pydantic-settings to load config from environment variables (.env file).

AWS credentials are deliberately absent: boto3 resolves them from the standard
credential chain — `aws configure` in development, the IAM execution role on
Lambda/ECS. Keeping keys out of app config means there is no field for a real
key to accidentally land in.
"""

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load .env into os.environ. pydantic-settings reads .env into Settings below,
# but it does not populate os.environ — and LangSmith tracing reads the
# LANGSMITH_* vars straight from os.environ. Without this, tracing stays off.
load_dotenv()


class Settings(BaseSettings):
    """Application settings — loaded from environment variables."""

    # AWS
    aws_default_region: str = Field(default="us-east-1")

    # Provider
    llm_provider: str = Field(
        default="auto",
        description=(
            "auto | bedrock | ollama | fake. 'auto' picks bedrock when boto3 can "
            "resolve credentials, otherwise ollama. 'fake' must be set "
            "explicitly: it pairs a canned answer with deterministic keyword "
            "embeddings so the retrieval path, tracing and evals all run with no "
            "model and no network."
        ),
    )

    # Generation model
    bedrock_model_id: str = Field(
        default="anthropic.claude-3-sonnet-20240229-v1:0",
        description="Amazon Bedrock model ID for the summarisation step",
    )
    bedrock_max_tokens: int = Field(default=768)
    # 0.1 rather than the 0.3 a first-response generator would use. This model's
    # job is to restate what the retrieved tickets say; sampling variety here
    # shows up as invention, which is the failure mode faithfulness measures.
    bedrock_temperature: float = Field(default=0.1)
    ollama_model: str = Field(default="llama3.2")

    # Embedding model
    #
    # Changing either of these invalidates an existing index. The vectors already
    # in Chroma were produced by the old model, and a query embedded by the new
    # one is compared against them as if they shared a space — they do not. The
    # result is not an error, it is quietly wrong retrieval, which is worse.
    # `python -m app.ingest --rebuild` after any change here.
    bedrock_embedding_model_id: str = Field(default="amazon.titan-embed-text-v2:0")
    ollama_embedding_model: str = Field(default="nomic-embed-text")

    # Retrieval
    chroma_persist_dir: str = Field(default="./chroma_db")
    chroma_collection: str = Field(default="past_tickets")
    retrieval_top_k: int = Field(
        default=3,
        description=(
            "Documents pulled per query. 3 is deliberate: every retrieved ticket "
            "spends context budget, and a fourth marginally-similar ticket costs "
            "precision without adding an answer."
        ),
    )
    retrieval_score_threshold: float | None = Field(
        default=None,
        description=(
            "Minimum similarity for a document to be retrieved at all. Off by "
            "default, and deliberately not given a default number: the useful "
            "cutoff differs per embedding model, so a value tuned against Titan "
            "would quietly filter everything out under Ollama. This is the lever "
            "to reach for when RAGAS context precision is low — measure the "
            "score distribution on your own index first, then set it."
        ),
    )

    # Analytics log
    database_url: str = Field(
        default="sqlite:///./ptk_analytics.db",
        description=(
            "SQLAlchemy URL for the search log. SQLite by default so the file can "
            "be opened and queried by hand; point it at Postgres in production and "
            "no application code changes."
        ),
    )
    log_searches: bool = Field(
        default=True,
        description=(
            "Whether to persist a row per search. Off for load testing, where the "
            "log measures the log."
        ),
    )

    # No judge settings here on purpose. The RAGAS judge runs in a separate
    # virtualenv where `app` is not importable (see requirements-evals.txt), so
    # it reads JUDGE_* straight from the environment. A field here would be dead
    # config that looks live. extra="ignore" below is what lets those vars sit in
    # the same .env without pydantic rejecting them.

    # API
    app_title: str = Field(default="Past Ticket Knowledge API")
    app_version: str = Field(default="1.0.0")
    log_level: str = Field(default="INFO")

    # extra="ignore": .env also holds vars consumed elsewhere (LANGSMITH_*, read
    # from os.environ by the tracer). Without this, pydantic rejects them.
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton — import this everywhere
settings = Settings()
