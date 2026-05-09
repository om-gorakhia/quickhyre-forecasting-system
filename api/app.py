"""FastAPI application factory."""

import logging
import time
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.routes import router
from api.data import store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load forecast artifacts on startup."""
    logger.info("Loading forecast artifacts...")
    try:
        store.load()
        logger.info("API ready — %d states loaded", len(store.states))
    except FileNotFoundError as e:
        logger.error("Failed to load artifacts: %s", e)
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Sales Forecasting API",
    description="8-week beverage sales forecasts for US states. "
                "Models: SARIMA, Prophet, XGBoost, LSTM with automatic best-model selection.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Middleware: request logging ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    elapsed = (time.time() - t0) * 1000
    logger.info("%s %s → %d (%.1fms)", request.method, request.url.path, response.status_code, elapsed)
    return response


# --- Global exception handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": str(exc)},
    )


# Mount routes under /api/v1
app.include_router(router, prefix="/api/v1")
