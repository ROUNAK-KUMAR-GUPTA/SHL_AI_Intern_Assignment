# Conversational SHL Assessment Recommender

A FastAPI-based conversational agent that recommends SHL Individual Test Solutions based on hiring needs.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python run.py

# Or with custom host/port
HOST=0.0.0.0 PORT=8000 python run.py
```

## API Endpoints

### GET /health
Returns service health status.

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

### POST /chat
Send conversation history and receive agent reply with recommendations.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I am hiring for a graduate trainee position and need cognitive, personality, and situational judgement assessments."}
    ]
  }'
```

**Request Schema:**
```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."}
  ]
}
```

**Response Schema:**
```json
{
  "reply": "string",
  "recommendations": [
    {
      "name": "string",
      "url": "string",
      "test_type": "string"
    }
  ],
  "end_of_conversation": false
}
```

## Conversational Behaviors

| Behavior | Description |
|----------|-------------|
| **Clarify** | Asks follow-up questions when the query is vague (no specifics about role, technologies, or measurement types) |
| **Recommend** | Returns 1-10 assessments with names and URLs from the SHL catalog |
| **Refine** | Handles add/remove modifications to the existing shortlist |
| **Compare** | Provides grounded comparisons between two assessments using catalog data |

## Project Structure

```
├── app/
│   ├── __init__.py          # Package init
│   ├── main.py              # FastAPI app with /health and /chat endpoints
│   ├── agent.py             # Conversational agent pipeline
│   ├── catalog_loader.py    # Catalog loading, search (TF-IDF + FAISS), and formatting
│   └── recommender.py       # Recommendation engine (tech keywords, role templates, scoring)
├── catalog_clean.json       # Cleaned SHL product catalog (377 items)
├── requirements.txt         # Python dependencies
├── run.py                   # Server runner
├── test_api.py              # Comprehensive test suite
├── approach_document.md     # 2-page approach document
└── README.md                # This file
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `LLM_API_KEY` | `""` | OpenAI-compatible API key (optional) |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | LLM API base URL |
| `LLM_MODEL` | `gpt-4o-mini` | LLM model name |

### Without LLM
The system works fully without an LLM using rule-based fallbacks for all natural language generation.

## Testing

```bash
# Start the server first
python run.py &

# Run the test suite
python test_api.py
```

Tests cover: health check, all sample conversations (C1-C10), off-topic refusal, legal question redirection, comparison handling, modification handling, prompt injection resistance, and schema compliance.

## Design Decisions

1. **Hybrid recommendation**: Combines rule-based logic (tech keywords, role templates) with TF-IDF semantic search for best recall
2. **Stateless design**: All conversation state is in the request payload; no server-side session storage
3. **Catalog-first URLs**: Every URL in recommendations comes from the scraped catalog—no hallucinations
4. **Graceful LLM fallback**: Works with or without an LLM API key
5. **Seniority-aware filtering**: Entry-level items are penalized/filtered for senior/executive queries
