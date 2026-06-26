"""
Locust load test for the Financial RAG Engine.

Target: GET /api/v1/retrieve-only (no LLM, pure retrieval latency)
Users: 20 concurrent
Ramp-up: 2 users/second
Duration: 5 minutes
p95 target: < 400ms

Run with:
    locust -f tests/load/locustfile.py --headless \\
        -u 20 -r 2 --run-time 5m \\
        --html results/load_test_report.html \\
        --host http://localhost:5000
"""

import random

from locust import HttpUser, between, task

# ---------------------------------------------------------------------------
# 20 diverse financial questions for load testing
# ---------------------------------------------------------------------------

FINANCIAL_QUESTIONS = [
    "What was Apple's total revenue in Q4 2023?",
    "What was Apple's iPhone revenue in Q4 fiscal year 2023?",
    "What was Apple's Services revenue record in 2023?",
    "What was Apple's gross margin in Q4 2023?",
    "What did Tim Cook say about artificial intelligence?",
    "What guidance did Luca Maestri give for Q1 2024?",
    "What was Microsoft's total revenue for fiscal year 2023?",
    "How much did Azure grow year over year in 2023?",
    "What was Microsoft 365 Commercial revenue growth?",
    "What was Microsoft's operating income in FY2023?",
    "What did Satya Nadella say about AI integration?",
    "How many vehicles did Tesla deliver in Q3 2023?",
    "What was Tesla's total revenue in Q3 2023?",
    "What was Tesla's automotive gross margin in Q3 2023?",
    "What was Tesla's energy generation and storage revenue?",
    "What did Elon Musk say about autonomous driving?",
    "When are Cybertruck deliveries expected to begin?",
    "What was Tesla's cash position at end of Q3 2023?",
    "What was Apple's Greater China revenue in Q4 2023?",
    "What was Microsoft's Intelligent Cloud revenue FY2023?",
]

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import itertools
import threading

# Rotate through all CLIENT_API_KEYS defined in environment
CLIENT_KEYS_STR = os.getenv("CLIENT_API_KEYS", "client_key_1,client_key_2")
CLIENT_KEYS = [k.strip() for k in CLIENT_KEYS_STR.split(",") if k.strip()]

if CLIENT_KEYS:
    CLIENT_KEYS_CYCLE = itertools.cycle(CLIENT_KEYS)
else:
    CLIENT_KEYS_CYCLE = itertools.cycle(["client_key_1"])

CLIENT_KEYS_LOCK = threading.Lock()


def get_next_api_key() -> str:
    """Get the next API key in a thread-safe manner."""
    with CLIENT_KEYS_LOCK:
        return next(CLIENT_KEYS_CYCLE)


class RAGUser(HttpUser):
    """
    Simulated user hitting the retrieve-only endpoint.

    Each user picks a random question from the pool and fires
    GET /api/v1/retrieve-only repeatedly.
    """

    wait_time = between(0.5, 2.0)  # 0.5-2s think time between requests

    def on_start(self) -> None:
        """Assign an API key to this virtual user."""
        self.api_key = get_next_api_key()

    @task(10)
    def retrieve_only(self) -> None:
        """
        Send a retrieval-only request (no LLM).

        This measures pure retrieval latency: embed + FAISS.
        """
        question = random.choice(FINANCIAL_QUESTIONS)
        params = {
            "question": question,
            "top_k": 5,
            "mode": "hybrid",
        }
        headers = {"X-API-Key": self.api_key}

        with self.client.get(
            "/api/v1/retrieve-only",
            params=params,
            headers=headers,
            name="/api/v1/retrieve-only",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "chunks" not in data:
                    response.failure("Missing 'chunks' field in response")
                else:
                    response.success()
            elif response.status_code == 429:
                response.failure("Rate limit hit during load test")
            else:
                response.failure(f"Unexpected status code: {response.status_code}")

    @task(1)
    def health_check(self) -> None:
        """
        Occasionally hit the health endpoint to verify service liveness.
        """
        with self.client.get(
            "/api/v1/health",
            name="/api/v1/health",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
