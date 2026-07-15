# wsgi.py — Gunicorn entry point for production
# Usage: gunicorn --workers 2 --bind 0.0.0.0:8000 wsgi:application
from app import app as application

if __name__ == "__main__":
    application.run()
