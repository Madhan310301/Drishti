"""
FastAPI Main Application Entrypoint for Drishti Predictive Command Console.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.common.logger import get_logger
from backend.common.constants import PROJECT_ROOT
from backend.database.connection import init_db
from backend.api.routes import router

logger = get_logger(__name__)

APP_DIR = PROJECT_ROOT / "app"
INDEX_HTML = APP_DIR / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager for startup and shutdown events.
    """
    logger.info("Initializing Drishti API server...")
    try:
        init_db()
        logger.info("Database schema verified.")
    except Exception as exc:
        logger.error(f"Error during database startup initialization: {exc}")

    yield
    logger.info("Shutting down Drishti API server...")


app = FastAPI(
    title="Drishti – Predictive Command Console API",
    description="Backend Predictive Analytics & ETL Engine for Karnataka Police Command Console.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Router
app.include_router(router)

# Mount static files if app directory exists
if APP_DIR.exists():
    app.mount("/static", StaticFiles(directory=APP_DIR), name="static")


@app.get("/", include_in_schema=False)
@app.get("/dashboard", include_in_schema=False)
def serve_dashboard() -> FileResponse:
    """Serve the frontend dashboard web application."""
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML)
    return FileResponse(PROJECT_ROOT / "README.md")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=True)
