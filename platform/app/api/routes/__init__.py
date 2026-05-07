"""Versioned platform API routes package.

File summary:
- Groups all `/api/v1/...` route modules for the platform service.
- Keeps prediction, drift, registry, and promotion endpoints in separate files.
- Lets `app/main.py` register feature routers in one obvious place.
"""
