"""Platform service layer package.

File summary:
- Groups business logic used by FastAPI route handlers.
- Keeps API routes thin by moving prediction, drift, registry, promotion, and webhook work here.
- Separates orchestration logic from database models and ML helper functions.
"""
