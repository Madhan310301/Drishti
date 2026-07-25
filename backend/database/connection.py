"""
Database connection and session configuration for Drishti.
Supports PostgreSQL (Supabase compatible) with SQLite fallback.
"""

from typing import Generator
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from backend.common.logger import get_logger
from backend.common.constants import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

logger = get_logger(__name__)

# Fetch database URL from environment or default to local SQLite
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    db_path = PROJECT_ROOT / "drishti.db"
    DATABASE_URL = f"sqlite:///{db_path}"
    logger.info(f"No DATABASE_URL found in env. Falling back to local SQLite: {DATABASE_URL}")
else:
    logger.info(f"Connected to database engine specified in DATABASE_URL")

# Create engine with sqlite connect_args if applicable
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args, echo=False)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def init_db() -> None:
    """
    Initialize database schema tables.
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully.")
    except Exception as exc:
        logger.error(f"Error initializing database schema: {exc}")
        raise


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a database session context per request.

    Yields
    ------
    Session
        SQLAlchemy database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
