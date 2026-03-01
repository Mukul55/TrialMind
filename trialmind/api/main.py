"""
FastAPI application entry point for TrialMind.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

app = FastAPI(
    title="TrialMind API",
    description="Clinical Trial Protocol Optimization RAG System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Module-level singletons (initialized on startup)
vector_store = None
analyzer = None


@app.on_event("startup")
async def startup_event():
    global vector_store, analyzer

    logger.info("Initializing TrialMind vector store...")

    try:
        from processing.embedder import TrialMindVectorStore
        from synthesis.protocol_analyzer import ProtocolAnalyzer

        vector_store = TrialMindVectorStore()
        analyzer = ProtocolAnalyzer(vector_store)
        logger.info("TrialMind API ready")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        logger.warning(
            "API started with errors. Ensure dependencies are installed "
            "and ANTHROPIC_API_KEY is set in .env"
        )


# Register routes
from api.routes import router
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
