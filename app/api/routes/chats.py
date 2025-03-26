"""
Chat API endpoints for conversations and messages.
"""
import logging
import json
import uuid
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse
import asyncio

from app.db.database import get_db
from app.db.models import Conversation, Message, Repository, User
from app.api.schemas.chat import (
    ConversationCreate, 
    ConversationResponse, 
    MessageCreate, 
    MessageResponse,
    ChatRequest
)
from app.api.dependencies import get_current_user
from app.rag.advanced_rag import AdvancedRAG
from app.vector_store.chroma_store import ChromaStore
from app.utils.llm import LLMClient

router = APIRouter()
logger = logging.getLogger(__name__)

# Get RAG components
store = ChromaStore()
llm_client = LLMClient()
rag = AdvancedRAG(store, llm_client)

@router.post("/conversations", response_model=ConversationResponse)
def create_conversation(
    conversation: ConversationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new conversation."""
    # Check if repository exists and user has access
    repository = db.query(Repository).filter(
        Repository.id == conversation.repository_id
    ).first()
    
    if not repository:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository not found"
        )
    
    # Check if user has access to repository (owner or public)
    if repository.owner_id != current_user.id and not repository.is_public:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this repository"
        )
    
    # Create conversation
    db_conversation = Conversation(
        id=uuid.uuid4(),
        user_id=current_user.id,
        repository_id=conversation.repository_id,
        title=conversation.title
    )
    
    db.add(db_conversation)
    db.commit()
    db.refresh(db_conversation)
    
    return db_conversation

@router.get("/conversations", response_model=List[ConversationResponse])
def get_conversations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    repository_id: Optional[uuid.UUID] = None
):
    """Get all conversations for the current user."""
    query = db.query(Conversation).filter(Conversation.user_id == current_user.id)
    
    if repository_id:
        query = query.filter(Conversation.repository_id == repository_id)
    
    return query.order_by(Conversation.updated_at.desc()).all()

@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
def get_conversation(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a specific conversation."""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    return conversation

@router.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
def get_messages(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all messages for a conversation."""
    # Check if conversation exists and user has access
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    return db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.created_at.asc()).all()

@router.post("/conversations/{conversation_id}/messages")
async def create_message(
    conversation_id: uuid.UUID,
    message: MessageCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new message in a conversation with streaming response."""
    # Check if conversation exists and user has access
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Create user message
    user_message = Message(
        id=uuid.uuid4(),
        conversation_id=conversation_id,
        is_user=True,
        content=message.content,
        created_at=datetime.utcnow()
    )
    
    db.add(user_message)
    db.commit()
    
    # Update conversation's updated_at
    conversation.updated_at = datetime.utcnow()
    db.commit()
    
    # Get conversation history
    messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.created_at.asc()).all()
    
    conversation_history = [
        {"query": msg.content, "answer": next_msg.content if i+1 < len(messages) else ""}
        for i, msg in enumerate(messages[::2])  # Every other message (user messages)
        if i+1 < len(messages)  # Make sure there's a system response
    ]
    
    # Create empty system message to attach the response to
    system_message = Message(
        id=uuid.uuid4(),
        conversation_id=conversation_id,
        is_user=False,
        content="",  # Will be populated as the response streams
        created_at=datetime.utcnow()
    )
    
    db.add(system_message)
    db.commit()
    
    # Use SSE for streaming
    return EventSourceResponse(
        generate_rag_response(
            message.content, 
            conversation_history, 
            system_message.id,
            db
        )
    )

async def generate_rag_response(query, conversation_history, message_id, db):
    """Generate streaming response from RAG."""
    try:
        # Get the message from DB
        db_message = db.query(Message).filter(Message.id == message_id).first()
        
        # Process query
        response = rag.query(query, conversation_history)
        answer = response["answer"]
        
        # Stream the response word by word
        chunks = answer.split(" ")
        full_response = ""
        
        for i, chunk in enumerate(chunks):
            full_response += chunk + " "
            
            # Update the message in the database
            db_message.content = full_response.strip()
            db.commit()
            
            # Yield the chunk for streaming
            yield {
                "event": "message", 
                "data": json.dumps({
                    "id": str(message_id),
                    "content": chunk + " " if i < len(chunks) - 1 else chunk,
                    "is_complete": False
                })
            }
            
            # Small delay for smoother streaming
            await asyncio.sleep(0.05)
        
        # Include code snippets and explanations in the metadata if available
        metadata = {}
        if response.get("code_snippets"):
            metadata["code_snippets"] = response["code_snippets"]
        
        if response.get("explanations"):
            metadata["explanations"] = [
                {
                    "explanation": e.explanation if hasattr(e, "explanation") else e, 
                    "best_practices": e.best_practices if hasattr(e, "best_practices") else []
                } 
                for e in response["explanations"]
            ]
        
        # Update the message metadata
        db_message.message_metadata = metadata
        db.commit()
        
        # Signal completion
        yield {
            "event": "message", 
            "data": json.dumps({
                "id": str(message_id),
                "is_complete": True,
                "metadata": metadata
            })
        }
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        yield {
            "event": "error", 
            "data": json.dumps({
                "error": str(e)
            })
        }

@router.post("/stream", response_class=EventSourceResponse)
async def stream_chat(
    chat_request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Stream a chat response without creating a persistent conversation."""
    async def event_generator():
        try:
            # Process query
            response = rag.query(chat_request.query, chat_request.history or [])
            answer = response["answer"]
            
            # Stream the response word by word
            chunks = answer.split(" ")
            
            for i, chunk in enumerate(chunks):
                # Yield the chunk for streaming
                yield {
                    "event": "message", 
                    "data": json.dumps({
                        "content": chunk + " " if i < len(chunks) - 1 else chunk,
                        "is_complete": False
                    })
                }
                
                # Small delay for smoother streaming
                await asyncio.sleep(0.05)
            
            # Include code snippets and explanations in the metadata if available
            metadata = {}
            if response.get("code_snippets") and chat_request.include_snippets:
                metadata["code_snippets"] = response["code_snippets"]
            
            if response.get("explanations") and chat_request.explain_code:
                metadata["explanations"] = [
                    {
                        "explanation": e.explanation if hasattr(e, "explanation") else e, 
                        "best_practices": e.best_practices if hasattr(e, "best_practices") else []
                    } 
                    for e in response["explanations"]
                ]
            
            # Signal completion
            yield {
                "event": "message", 
                "data": json.dumps({
                    "is_complete": True,
                    "metadata": metadata
                })
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            yield {
                "event": "error", 
                "data": json.dumps({
                    "error": str(e)
                })
            }
    
    return EventSourceResponse(event_generator()) 