"""FastAPI Server exposing AgentX endpoints."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from agentx.config import settings
from agentx.models.schemas import AgentResponse, TaskRequest
from agentx.orchestrator.orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)


def create_app(orchestrator: Optional[AgentOrchestrator] = None) -> FastAPI:
    """Factory creating FastAPI application for AgentX."""
    app = FastAPI(
        title="AgentX - Multi-Agent Collaboration Framework",
        description="FastAPI service enabling specialized legal AI agents to collaboratively solve complex tasks.",
        version="0.1.0",
    )

    # In-memory registry of executed tasks for status querying
    task_results: Dict[str, Dict[str, Any]] = {}
    active_orchestrator = orchestrator or AgentOrchestrator()

    # Error Handling per LLD Section 5: "Invalid API request -> Return HTTP 400 with error message"
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning("Validation error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "detail": "Invalid API request",
                "errors": exc.errors(),
            },
        )

    @app.get("/status", tags=["System"])
    async def get_system_status(task_id: Optional[str] = None):
        """GET /status endpoint.

        Returns overall system health, or specific task status if `task_id` is supplied.
        """
        if task_id:
            record = task_results.get(task_id)
            if not record:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Task '{task_id}' not found",
                )
            return record

        return {
            "status": "healthy",
            "framework": "AgentX Multi-Agent Collaboration Framework",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "registered_agents": active_orchestrator.agent_manager.list_agents(),
            "tasks_processed": len(task_results),
            "orchestrator_state": (
                active_orchestrator.state.status if active_orchestrator.state else "idle"
            ),
        }

    @app.post("/task", response_model=AgentResponse, tags=["Tasks"])
    async def create_task(request: TaskRequest):
        """POST /task endpoint.

        Receives TaskRequest payload, triggers orchestrator workflow, and returns AgentResponse.
        """
        # Validate task_id presence and structure per Section 5
        if not request.task_id or not request.task_id.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid API request: 'task_id' must not be empty",
            )

        task_id = request.task_id.strip()

        # Record task submission
        task_results[task_id] = {
            "task_id": task_id,
            "status": "processing",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            # Execute workflow via AgentOrchestrator
            response: AgentResponse = await active_orchestrator.arun_workflow(request)

            task_results[task_id] = {
                "task_id": task_id,
                "status": response.status,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "output_data": response.output_data,
                "errors": response.errors,
            }
            return response
        except Exception as e:
            logger.error("Error executing task %s: %s", task_id, e, exc_info=True)
            task_results[task_id] = {
                "task_id": task_id,
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "errors": [str(e)],
            }
            return AgentResponse(
                task_id=task_id,
                output_data={},
                status="failed",
                errors=[str(e)],
            )

    @app.get("/task/{task_id}", response_model=Dict[str, Any], tags=["Tasks"])
    async def get_task_details(task_id: str):
        """Retrieve execution details for a specific task."""
        record = task_results.get(task_id)
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task '{task_id}' not found",
            )
        return record

    return app


# Default application instance for uvicorn
app = create_app()
