"""Main FastAPI application for the translation system.

This module creates and configures the FastAPI application with:
- API routes from app.api.routes
- Error handlers
- Middleware
- CORS configuration
- Lifespan events (startup/shutdown)
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Load environment variables from .env file
# This must happen before importing other app modules that might use env vars
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)

from app.api.routes import router as api_router
from app.api.routes import setup_error_handlers
from app.utils.logger import get_logger

# Initialize logger
logger = get_logger("app.main")

# Track application start time for health checks
_app_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events (startup and shutdown).
    
    This context manager handles:
    - Startup: Initialize models, connections, and resources
    - Shutdown: Clean up resources and close connections
    
    Args:
        app: The FastAPI application instance
    """
    global _app_start_time
    
    # STARTUP
    logger.info("=" * 50)
    logger.info("Translation System API Starting...")
    logger.info("=" * 50)
    
    _app_start_time = time.time()
    
    # Log configuration info
    logger.info(f"Environment loaded from: {env_path}")
    logger.info(f"API documentation available at: /docs")
    
    # You can add initialization logic here:
    # - Load ML models eagerly (instead of lazy loading)
    # - Connect to databases
    # - Initialize caches
    # - Warm up inference pipelines
    
    logger.info("Application startup complete")
    
    yield  # Application runs here
    
    # SHUTDOWN
    logger.info("=" * 50)
    logger.info("Translation System API Shutting down...")
    logger.info("=" * 50)
    
    uptime = time.time() - _app_start_time
    logger.info(f"Total uptime: {uptime:.2f} seconds")
    
    # Clean up resources here:
    # - Close database connections
    # - Release model resources
    # - Flush logs
    # - Close cache connections
    
    logger.info("Application shutdown complete")


def create_application() -> FastAPI:
    """Create and configure the FastAPI application.
    
    This factory function creates a fully configured FastAPI app with:
    - Title and version metadata
    - Lifespan event handlers
    - CORS middleware
    - API routes
    - Error handlers
    
    Returns:
        Configured FastAPI application instance
    """
    # Create the FastAPI app
    app = FastAPI(
        title="Translation System API",
        description="""
        A comprehensive translation system providing:
        
        - **Audio Transcription (ASR)**: Convert speech to text using Whisper
        - **Text Translation**: Translate between languages using NLLB
        - **Text-to-Speech (TTS)**: Synthesize speech using Piper
        - **Pipeline Processing**: Chain multiple services together
        
        ## Pipeline Types
        
        1. **transcribe_translate**: Audio → Text → Translated Text
        2. **translate_synthesize**: Text → Translated Text → Audio
        3. **full**: Audio → Text → Translated Text → Audio
        
        ## Authentication
        
        Some endpoints may require authentication (configure via environment variables).
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    
    # Add CORS middleware to allow cross-origin requests
    # In production, restrict this to your actual frontend domain
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Change to specific domains in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add request timing middleware
    @app.middleware("http")
    async def add_request_timing(request: Request, call_next):
        """Middleware to track request processing time."""
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    
    # Include API routes
    app.include_router(api_router)
    
    # Set up error handlers
    setup_error_handlers(app)
    
    return app


# Create the application instance
# This is imported by ASGI servers like uvicorn
app = create_application()


# Root endpoint (outside the API router)
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information.
    
    Returns:
        Welcome message with API documentation links
    """
    return {
        "message": "Welcome to the Translation System API!",
        "version": "1.0.0",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
        },
        "endpoints": {
            "health": "/api/v1/health",
            "ready": "/api/v1/ready",
            "transcribe": "/api/v1/transcribe",
            "translate": "/api/v1/translate",
            "synthesize": "/api/v1/synthesize",
            "pipeline": "/api/v1/pipeline",
        },
    }
