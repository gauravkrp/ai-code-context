"""
FastAPI application for the AI Code Context backend.
"""
import logging
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.db.database import get_db, init_db
from app.config.settings import config

# Initialize logging
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Code Context API",
    description="API for interacting with the AI Code Context system",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers
from app.api.routes import auth, repositories, chats, users

app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(repositories.router, prefix="/api/repositories", tags=["Repositories"])
app.include_router(chats.router, prefix="/api/chats", tags=["Chats"])
app.include_router(users.router, prefix="/api/users", tags=["Users"])

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized.")

@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": app.version} 