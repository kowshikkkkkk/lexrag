import random
from locust import HttpUser, task, between


# Sample legal queries to simulate real user behavior
LEGAL_QUERIES = [
    "what is theft under IPC",
    "what is the punishment for murder",
    "what is culpable homicide",
    "define cheating under IPC",
    "what is the punishment for rape",
    "what is dowry death",
    "define abetment under IPC",
    "what is criminal conspiracy",
    "what is extortion under IPC",
    "what is wrongful confinement",
    "punishment for robbery under IPC",
    "what is hurt under IPC",
    "define grievous hurt",
    "what is defamation under IPC",
    "punishment for forgery under IPC",
]


class LexRAGUser(HttpUser):
    """
    Simulates a real user querying LexRAG.
    Waits 1-3 seconds between requests — realistic think time.
    """
    wait_time = between(1, 3)
    host = "http://localhost:8000"

    @task(7)
    def query(self):
        """Standard query — most common operation (weight 7)."""
        query = random.choice(LEGAL_QUERIES)
        self.client.post(
            "/api/v1/query",
            json={
                "query": query,
                "rewrite": True,
            },
            name="/api/v1/query",
        )

    @task(2)
    def cached_query(self):
        """
        Repeat query — tests cache hit rate (weight 2).
        Uses fixed query so Redis returns it instantly.
        """
        self.client.post(
            "/api/v1/query",
            json={
                "query": "what is theft under IPC",
                "rewrite": True,
            },
            name="/api/v1/query [cached]",
        )

    @task(1)
    def health_check(self):
        """Health check — lightweight (weight 1)."""
        self.client.get("/api/v1/health", name="/api/v1/health")

    def on_start(self):
        """Called when a virtual user starts."""
        # Warm up with a health check
        self.client.get("/api/v1/health")