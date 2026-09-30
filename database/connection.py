import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Check Streamlit Cloud secrets first, then environment variable, then local default
DATABASE_URL = None
try:
    import streamlit as st
    if hasattr(st, "secrets") and "DATABASE_URL" in st.secrets:
        DATABASE_URL = st.secrets["DATABASE_URL"]
except Exception:
    pass

if not DATABASE_URL:
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/churn_db")

# Normalize legacy postgres:// URI scheme to postgresql:// for SQLAlchemy compatibility
if DATABASE_URL:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

    # Automatically adapt driver to installed DBAPI packages (psycopg2 vs psycopg)
    has_psycopg = False
    try:
        import psycopg  # noqa: F401
        has_psycopg = True
    except ImportError:
        pass

    has_psycopg2 = False
    try:
        import psycopg2  # noqa: F401
        has_psycopg2 = True
    except ImportError:
        pass

    # If postgresql+psycopg:// was specified but only psycopg2 is installed
    if DATABASE_URL.startswith("postgresql+psycopg://") and not has_psycopg and has_psycopg2:
        DATABASE_URL = DATABASE_URL.replace("postgresql+psycopg://", "postgresql+psycopg2://", 1)
    # If postgresql+psycopg2:// was specified but only psycopg v3 is installed
    elif DATABASE_URL.startswith("postgresql+psycopg2://") and not has_psycopg2 and has_psycopg:
        DATABASE_URL = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql+psycopg://", 1)
    # If generic postgresql:// without driver was specified, pick the installed driver
    elif DATABASE_URL.startswith("postgresql://"):
        if has_psycopg2:
            DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)
        elif has_psycopg:
            DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

# Connection parameters for local and serverless cloud PostgreSQL (e.g. Neon, Supabase)
connect_args = {}
if "postgresql" in DATABASE_URL:
    connect_args["connect_timeout"] = 5

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
