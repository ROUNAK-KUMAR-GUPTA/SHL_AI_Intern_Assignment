# Approach Document: Conversational SHL Assessment Recommender

## Overview

This project implements a stateless conversational agent that recommends SHL Individual Test Solutions based on hiring needs. The system combines rule-based logic with TF-IDF semantic search to deliver accurate, grounded recommendations across four conversational behaviors: Clarify, Recommend, Refine, and Compare.

## Architecture

The application is built as a FastAPI service with two endpoints: `GET /health` for readiness checks and `POST /chat` for conversational interaction. All state is carried in the request payload (conversation history), making the service fully stateless and horizontally scalable. The agent pipeline processes each message through a series of checks: off-topic detection, legal question redirection, comparison handling, modification detection, vagueness clarification, and finally recommendation generation.

## Data Pipeline

The SHL product catalog was scraped and cleaned into a structured JSON file (377 Individual Test Solutions). Each entry includes the assessment name, URL, test type codes (K/P/A/B/S/C/D/E derived from product keys), job levels, languages, duration, and description. Descriptions were enriched with search aliases (e.g., "coding" → "programming", "devops" → "docker kubernetes") to improve TF-IDF recall. A FAISS index built over TF-IDF vectors provides fast semantic search across the catalog.

## Recommendation Engine

The recommendation engine employs four strategies in priority order. First, direct technology keyword matching maps explicit terms like "Java", "Docker", "SQL", or "AWS" to known assessments with a high score boost. Second, role-based templates provide curated shortlists for common roles (developer, senior leader, safety, contact center, graduate, finance, admin, sales, healthcare) with always-include and optional items. Third, TF-IDF semantic search catches requirements that don't map to predefined keywords. Fourth, measurement type filtering ensures cognitive, personality, or situational judgement requirements are met.

Results are scored and ranked using multiple signals: technology match bonus (+25), template inclusion bonus, seniority match, measurement type alignment, core assessment bonus, and penalties for entry-level items appearing in senior queries (-30) or solution-level items. A relevance filter removes items with only stop-word overlap. The top 10 items are returned as structured recommendations.

## Conversational Behaviors

**Clarify**: On the first turn, if the query is vague (lacks specifics like technologies, measurement types, or role+seniority context), the agent asks targeted clarification questions rather than recommending. Role-specific clarifications are provided for senior leadership (selection vs. development) and contact center (language requirements).

**Recommend**: When enough context is available, the engine commits to a shortlist of 1–10 assessments with names and URLs. Every URL is sourced from the scraped catalog—no hallucinations.

**Refine**: Modification requests (add/remove) are detected via regex patterns. The agent extracts the previous shortlist from conversation history URLs, applies modifications with fuzzy name matching (stripping articles, matching acronyms like "OPQ" to "OPQ32r"), and returns the updated shortlist.

**Compare**: When users ask to compare assessments, the agent retrieves both items from the catalog and produces a grounded comparison covering test type, duration, languages, and key differences.

## Safety Guards

The agent refuses off-topic requests (poems, recipes, general questions), redirects legal/compliance questions to appropriate teams, and resists prompt injection attempts. All responses strictly conform to the required schema: `{"reply": str, "recommendations": [...], "end_of_conversation": bool}`.

## LLM Integration (Optional)

An OpenAI-compatible LLM can be configured via environment variables (`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`) for natural language reply generation. When no LLM is available, the system falls back to rule-based templates that produce clear, structured responses with markdown tables. This ensures the system works fully without external API dependencies.

## Tech Stack

- **Framework**: FastAPI with Pydantic validation
- **Search**: scikit-learn TF-IDF + FAISS index
- **Data**: 377 cleaned SHL Individual Test Solutions
- **Runtime**: Uvicorn ASGI server, Python 3.11
- **Optional**: OpenAI-compatible LLM for NLG

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Returns `{"status": "ok"}` with HTTP 200 |
| `/chat` | POST | Accepts conversation history, returns agent reply with recommendations |

## Deployment

The service runs on port 8000 and is accessible at the public endpoint. All catalog data is loaded at startup (~0.1s), and each chat request completes within the 30-second timeout requirement.
