"""Backbone of the translation system, responsible 
for orchestrating the various components."""

from pathlib import Path
from dotenv import load_dotenv

# Load project/.env no matter the current working directory
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# FastAPI provides the web app and the Response/status helpers let us set HTTP codes.
from fastapi import FastAPI, Response, status
from app.utils.logger import get_logger
# asyncio is used for timeouts on readiness checks, and time keeps uptime for /health.
import asyncio
import time

# Create the application object and shared logger once when the module loads.
app = FastAPI()
logger = get_logger("app")
# Capture the startup time so /health can report uptime.
_start_time = time.time()


# Temporary readiness checks. These will later be replaced with real DB, cache, and model checks.
async def _check_db() -> bool:
    # replace with real async DB ping when ready
    return True

async def _check_cache() -> bool:
    # replace with real cache ping
    return True

async def _check_model() -> bool:
    # replace with model-loaded check
    return True


# Root route: gives a simple welcome message so you can confirm the API is responding.
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Translation System API!"}

# Health route: tells you the app process is alive and how long it has been running.
@app.get("/health")
async def health():
    uptime = time.time() - _start_time
    payload = {"status": "ok", "uptime_seconds": round(uptime, 2)}
    logger.info("Health check OK")
    return payload

# Readiness route: checks whether important pieces are ready before serving real traffic.
@app.get("/ready")
async def ready(response: Response):
    # Wrap each check in a timeout so one slow dependency does not freeze the request.
    try:
        db_ok = await asyncio.wait_for(_check_db(), timeout=5.0)
    except Exception:
        db_ok = False

    # Cache should respond quickly; if it does not, we mark it as not ready.
    try:
        cache_ok = await asyncio.wait_for(_check_cache(), timeout=5.0)
    except Exception:
        cache_ok = False

    # The model check tells you whether the translation model is available.
    try:
        model_ok = await asyncio.wait_for(_check_model(), timeout=5.0)
    except Exception:
        model_ok = False

    # Build a list of any failed checks so the response can say exactly what is wrong.
    failed = [name for name, ok in (("db", db_ok), ("cache", cache_ok), ("model", model_ok)) if not ok]

    # If any check failed, return 503 so orchestration tools know the service is not ready.
    if failed:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        logger.warning(f"Readiness failed: {failed}")
        return {"status": "unready", "failed": failed}

    # If every check passed, the service is ready to receive traffic.
    logger.info("Readiness check OK")
    return {"status": "ready"}