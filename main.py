"""Main entrypoint for AgentX Multi-Agent Collaboration Framework."""

import uvicorn
from agentx.config import settings


def main():
    """Runs the FastAPI server with uvicorn."""
    print("=" * 60)
    print("AgentX - Multi-Agent Collaboration Framework")
    print(f"Starting server on http://{settings.API_HOST}:{settings.API_PORT}")
    print("Endpoints:")
    print(f"  POST http://{settings.API_HOST}:{settings.API_PORT}/task")
    print(f"  GET  http://{settings.API_HOST}:{settings.API_PORT}/status")
    print(f"  Docs http://{settings.API_HOST}:{settings.API_PORT}/docs")
    print("=" * 60)

    uvicorn.run(
        "agentx.api.server:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
