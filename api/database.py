import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# En production (Render), DATABASE_URL pointe vers PostgreSQL.
# En local, on replie sur SQLite.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./db.sqlite3")

# Render renvoie parfois "postgres://" (obsolète), on corrige en "postgresql://"
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# SQLite nécessite check_same_thread=False; PostgreSQL non
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
