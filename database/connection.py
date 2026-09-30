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
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Connection parameters for local and serverless cloud PostgreSQL (e.g. Neon, Supabase)
connect_args = {}
if DATABASE_URL and DATABASE_URL.startswith("postgresql"):
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
