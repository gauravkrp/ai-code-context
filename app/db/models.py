"""
SQLAlchemy models for the application database.
"""
from datetime import datetime
from typing import List, Optional
import uuid

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class User(Base):
    """User model for authentication and permissions."""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(255), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    
    # Relationships
    repositories = relationship("Repository", back_populates="owner")
    conversations = relationship("Conversation", back_populates="user")
    
    def __repr__(self):
        return f"<User {self.username}>"

class Repository(Base):
    """Repository model for tracking indexed repositories."""
    __tablename__ = "repositories"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    url = Column(String(255), nullable=False)
    branch = Column(String(255), default="main")
    last_indexed = Column(DateTime, nullable=True)
    is_public = Column(Boolean, default=False)
    repo_metadata = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Unique constraint to prevent duplicate repositories
    __table_args__ = (UniqueConstraint('owner_id', 'url', 'branch', name='unique_repository'),)
    
    # Relationships
    owner = relationship("User", back_populates="repositories")
    conversations = relationship("Conversation", back_populates="repository")
    
    def __repr__(self):
        return f"<Repository {self.name} ({self.url})>"

class Conversation(Base):
    """Conversation model for tracking chat sessions."""
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    repository_id = Column(UUID(as_uuid=True), ForeignKey("repositories.id"), nullable=False)
    title = Column(String(255), default="Untitled Conversation")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    repository = relationship("Repository", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Conversation {self.title}>"

class Message(Base):
    """Message model for storing chat messages."""
    __tablename__ = "messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    is_user = Column(Boolean, default=True)  # True for user messages, False for system responses
    content = Column(Text, nullable=False)
    message_metadata = Column(JSONB, default=dict)  # Renamed from 'metadata' to avoid SQLAlchemy naming conflict
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    def __repr__(self):
        return f"<Message {'User' if self.is_user else 'System'} {self.created_at}>"

class VectorStoreLink(Base):
    """
    Links between database entities and vector store entries.
    This allows tracking which items in the database have vector representations.
    """
    __tablename__ = "vector_store_links"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_type = Column(String(255), nullable=False)  # 'repository', 'file', etc.
    entity_id = Column(UUID(as_uuid=True), nullable=False)  # ID of the entity in the database
    vector_id = Column(String(255), nullable=False)  # ID in the vector store
    collection_name = Column(String(255), nullable=False)
    link_metadata = Column(JSONB, default=dict)  # Renamed from 'metadata' to avoid SQLAlchemy naming conflict
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (UniqueConstraint('entity_type', 'entity_id', 'collection_name', name='unique_vector_link'),)
    
    def __repr__(self):
        return f"<VectorStoreLink {self.entity_type} {self.entity_id} -> {self.collection_name}/{self.vector_id}>" 