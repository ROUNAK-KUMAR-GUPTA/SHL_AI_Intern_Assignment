"""
FastAPI application for SHL Assessment Recommender.
Endpoints:
  GET /health - readiness check
  POST /chat - conversational assessment recommendation
"""
import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Literal

from .agent import process_chat, MAX_TURNS
from .catalog_loader import get_catalog

app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent that recommends SHL assessments based on user needs.",
    version="1.0.0",
)

@app.get("/")
async def root():
    return {
        "message": "Welcome to the SHL Assessment Recommender API",
        "status": "running",
        "health": "/health",
        "docs": "/docs"
    }

# Startup time for cold start tracking
_start_time = time.time()


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = []
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: str


@app.on_event("startup")
async def startup():
    """Pre-load catalog on startup."""
    get_catalog()
    print(f"Catalog loaded in {time.time() - _start_time:.1f}s")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a conversation and return the next agent reply.

    The API is stateless — every call carries the full conversation history.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    # Validate turn count
    turn_count = sum(1 for m in request.messages if m.role in ("user", "assistant"))
    if turn_count >= MAX_TURNS:
        return {
            "reply": "I've reached the maximum number of turns for this conversation. Here's my final shortlist based on what we've discussed.",
            "recommendations": [],
            "end_of_conversation": True
        }

    # Convert messages to dict format for agent
    messages_dicts = [{"role": m.role, "content": m.content} for m in request.messages]

    try:
        result = process_chat(messages_dicts)
    except Exception as e:
        # Graceful fallback
        catalog = get_catalog()
        search_query = " ".join(m["content"] for m in messages_dicts if m["role"] == "user")
        results = catalog.search(search_query, top_k=5)
        recommendations = [catalog.format_recommendation(item) for item in results]
        result = {
            "reply": f"I'd like to help you find the right assessments. Here are some that may be relevant to your needs.",
            "recommendations": recommendations,
            "end_of_conversation": False
        }

    return result
