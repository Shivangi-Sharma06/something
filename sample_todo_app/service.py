"""Todo service business logic with a seeded bug."""
from sample_todo_app.models import TodoStore, TodoItem

class TodoService:
    def __init__(self, store: TodoStore | None = None):
        self.store = store or TodoStore()

    def create_todo(self, title: str) -> TodoItem:
        return self.store.add(title)

    def mark_done(self, todo_id: int) -> TodoItem:
        return self.store.complete(todo_id)

    def get_summary(self) -> dict[str, int]:
        all_todos = self.store.list_all()
        total = len(all_todos)
        completed = sum(1 for t in all_todos if t.completed)
        # SEEDED BUG: Division by zero when there are no todos!
        completion_percentage = int((completed / total) * 100)
        return {
            "total": total,
            "completed": completed,
            "pending": total - completed,
            "completion_rate": completion_percentage,
        }
