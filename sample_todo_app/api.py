"""Todo API endpoints simulating a FastAPI/Flask service."""
from sample_todo_app.service import TodoService

service = TodoService()

def handle_create(title: str):
    return service.create_todo(title)

def handle_summary():
    # Delegates to service.get_summary()
    return service.get_summary()
