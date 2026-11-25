import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from backend.config import settings

logger = logging.getLogger(__name__)

db_url = settings.DATABASE_URL
# Handle postgres:// vs postgresql:// for SQLAlchemy compatibility on cloud platforms
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

connect_args = {}
if db_url.startswith("sqlite"):
    connect_args["check_same_thread"] = False

try:
    engine = create_engine(db_url, connect_args=connect_args)
    # Test connection
    with engine.connect() as conn:
        logger.info("Database connection established: %s", db_url.split("@")[-1] if "@" in db_url else db_url)
except Exception as e:
    logger.warning("Failed to connect to %s: %s. Falling back to local SQLite.", db_url, e)
    db_url = "sqlite:///./transcriptions.db"
    connect_args = {"check_same_thread": False}
    engine = create_engine(db_url, connect_args=connect_args)

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
    logger.info("Database tables initialized.")
