# Conversational SHL Assessment Recommender

A FastAPI-based conversational agent that recommends SHL assessments based on user requirements. The chatbot supports multi-turn conversations, recommends relevant SHL assessments, compares assessments, and refines recommendations based on follow-up user input.

---

## Live Demo

### Base URL
https://shl-ai-intern-assignment-9wf1.onrender.com

### Swagger API Documentation
https://shl-ai-intern-assignment-9wf1.onrender.com/docs

### Health Check
https://shl-ai-intern-assignment-9wf1.onrender.com/health

---

## Features

- Conversational SHL Assessment Recommendation
- Multi-turn conversation support
- SHL assessment recommendation
- Assessment comparison
- Recommendation refinement
- Health check endpoint
- Interactive Swagger UI
- REST API built with FastAPI
- Deployed on Render

---

## API Endpoints

### GET /health

Checks whether the API is running.

Example Response:

```json
{
  "status": "ok"
}
```

---

### POST /chat

Returns conversational assessment recommendations.

Example Request

```json
{
  "messages": [
    {
      "role": "user",
      "content": "I am hiring a Java developer."
    }
  ]
}
```

---

## Tech Stack

- Python
- FastAPI
- Uvicorn
- Pydantic
- NumPy
- Scikit-learn
- FAISS
- Render

---

## Project Structure

```
SHL_AI_Intern_Assignment/
│
├── app/
│   ├── main.py
│   ├── agent.py
│   ├── recommender.py
│   └── catalog_loader.py
│
├── sample_conversations/
├── catalog.json
├── catalog_clean.json
├── requirements.txt
├── run.py
├── README.md
└── approach_document.md
```

---

## Run Locally

Clone the repository

```bash
git clone https://github.com/ROUNAK-KUMAR-GUPTA/SHL_AI_Intern_Assignment.git
```

Go to the project folder

```bash
cd SHL_AI_Intern_Assignment
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python run.py
```

Open Swagger UI

```
http://127.0.0.1:8000/docs
```

---

## Deployment

This project is deployed on Render.

Base URL:

https://shl-ai-intern-assignment-9wf1.onrender.com

---

## Repository

GitHub Repository

https://github.com/ROUNAK-KUMAR-GUPTA/SHL_AI_Intern_Assignment

---

## Author

**Rounak Kumar Gupta**

GitHub:
https://github.com/ROUNAK-KUMAR-GUPTA
