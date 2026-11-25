import logging
import os
from typing import Optional
from urllib.parse import urlparse
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker
from backend.config import settings

logger = logging.getLogger(__name__)


def _redact_db_url(url: str) -> str:
    """Returns a non-sensitive representation of the DB URL without credentials."""
    if "@" in url:
        scheme = url.split("://", 1)[0] if "://" in url else "db"
        host_part = url.split("@", 1)[-1]
        return f"{scheme}://***@{host_part}"
    return url


def is_production_env() -> bool:
    """True when running in Render or explicit production environment."""
    return (
        os.getenv("RENDER", "").lower() == "true"
        or os.getenv("ENVIRONMENT", "").lower() == "production"
    )


def create_db_engine(raw_url: str, is_production: Optional[bool] = None):
    """
    Creates the SQLAlchemy engine and verifies connectivity at startup.
    In production with PostgreSQL configured, fails fast instead of falling back to SQLite.
    """
    prod = is_production_env() if is_production is None else is_production
    db_url = raw_url
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    safe_target = _redact_db_url(db_url)

    try:
        eng = create_engine(db_url, connect_args=connect_args, pool_pre_ping=True)
        with eng.connect():
            logger.info("Database connection established: %s", safe_target)
        return eng
    except Exception as e:
        if prod and db_url.startswith("postgresql://"):
            logger.error("Fatal: unable to connect to production PostgreSQL (%s): %s", safe_target, type(e).__name__)
            raise RuntimeError(f"Production PostgreSQL connection failed for {safe_target}") from e

        logger.warning("Failed to connect to %s (%s). Falling back to local SQLite.", safe_target, type(e).__name__)
        fallback_url = "sqlite:///./transcriptions.db"
        return create_engine(fallback_url, connect_args={"check_same_thread": False}, pool_pre_ping=True)


engine = create_db_engine(settings.DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    from backend import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    # Safe additive migration for existing databases lacking session_id
    inspector = inspect(engine)
    if "transcriptions" in inspector.get_table_names():
        cols = {c["name"] for c in inspector.get_columns("transcriptions")}
        if "session_id" not in cols:
            with engine.begin() as conn:
                conn.execute(
                    text("ALTER TABLE transcriptions ADD COLUMN session_id VARCHAR(64) NOT NULL DEFAULT 'default'")
                )
            logger.info("Migrated transcriptions table: added session_id column.")
    logger.info("Database tables initialized.")
