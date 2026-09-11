import os
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import bcrypt

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./secure_journal.db")

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class User(Base):
    __tablename__ = "users"
    
    # Using an anonymous unique hash ID to decouple user identities from application data
    id = Column(String, primary_key=True, index=True) 
    email_hashed = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    tos_accepted_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    entries = relationship("JournalEntry", back_populates="owner", cascade="all, delete-orphan")

class JournalEntry(Base):
    __tablename__ = "journal_entries"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id"), index=True, nullable=False)
    
    # Metadata payloads passed forward from Gate 2 processing layers
    escalation_score = Column(Integer, nullable=False)
    detected_framework_tags = Column(Text, nullable=False) # Stored as a serialized JSON string
    
    # Core entry text fields - Must be encrypted using app-level AES-256 keys in production
    encrypted_user_input = Column(Text, nullable=False) 
    encrypted_ai_response = Column(Text, nullable=False)
    
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    owner = relationship("User", back_populates="entries")

def init_db():
    Base.metadata.create_all(bind=engine)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def hash_password(plain_password: str) -> str:
    return bcrypt.hashpw(plain_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
