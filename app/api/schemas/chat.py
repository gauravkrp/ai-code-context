"""
Schemas for chat-related API endpoints.
"""
from typing import Any, Dict, List, Optional
import uuid
from datetime import datetime
from pydantic import BaseModel, Field


class ConversationBase(BaseModel):
    """Base schema for conversations."""
    title: str = "Untitled Conversation"


class ConversationCreate(ConversationBase):
    """Schema for creating a conversation."""
    repository_id: uuid.UUID


class ConversationResponse(ConversationBase):
    """Schema for conversation responses."""
    id: uuid.UUID
    user_id: uuid.UUID
    repository_id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    is_active: bool
    
    class Config:
        from_attributes = True


class MessageBase(BaseModel):
    """Base schema for messages."""
    content: str


class MessageCreate(MessageBase):
    """Schema for creating a message."""
    pass


class MessageResponse(MessageBase):
    """Schema for message responses."""
    id: uuid.UUID
    conversation_id: uuid.UUID
    is_user: bool
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    """Schema for chat stream requests."""
    query: str
    repository_id: Optional[uuid.UUID] = None
    history: Optional[List[Dict[str, str]]] = None
    include_snippets: bool = False
    explain_code: bool = False 