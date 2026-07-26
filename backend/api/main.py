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
from backend.etl.config import (
    HOTSPOT_CENTERS_FILE,
    HOTSPOT_MAP_HTML,
    NETWORK_GRAPH_HTML,
    SHAP_EXPLANATIONS_FILE,
)

logger = get_logger(__name__)

APP_DIR = PROJECT_ROOT / "app"
INDEX_HTML = APP_DIR / "index.html"


def _ensure_artifacts() -> None:
    """
    Generate ML artifacts on startup if they are missing. This makes a fresh
    `git clone` work with zero manual steps (the generated HTML/JSON files are
    gitignored, so they only exist after running the pipeline).
    """
    try:
        if not HOTSPOT_CENTERS_FILE.exists():
            from backend.ml.hotspots import main as hotspots_main
            hotspots_main()
        if not HOTSPOT_MAP_HTML.exists():
            from backend.ml.hotspot_map import HotspotMapBuilder
            HotspotMapBuilder().build()
        if not NETWORK_GRAPH_HTML.exists():
            from backend.ml.network_graph import NetworkGraphBuilder
            NetworkGraphBuilder().build()
        if not SHAP_EXPLANATIONS_FILE.exists():
            from backend.ml.explainability import ShapExplainer
            ShapExplainer().save()
        logger.info("Startup artifact check complete.")
    except Exception as exc:
        logger.error("Artifact auto-generation failed (dashboard maps may be empty): %s", exc)


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

    # Generate ML artifacts (maps, network, SHAP) if missing so a fresh clone works.
    _ensure_artifacts()

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

# Mount generated ML artifacts (maps, graphs) so the dashboard can embed them
OUTPUT_DATA_DIR = PROJECT_ROOT / "data" / "output"
if OUTPUT_DATA_DIR.exists():
    app.mount("/data/output", StaticFiles(directory=OUTPUT_DATA_DIR), name="output_data")


@app.get("/", include_in_schema=False)
@app.get("/dashboard", include_in_schema=False)
def serve_dashboard() -> FileResponse:
    """Serve the frontend dashboard web application."""
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML)
    return FileResponse(PROJECT_ROOT / "README.md")


if __name__ == "__main__":
    import os
    import uvicorn
    # PaaS platforms (Zoho Nimbus, Heroku, Render, ...) inject the listen port
    # via the PORT env var. Bind to it; never use reload in production.
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=port, reload=False)
