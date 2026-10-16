"""Todo models and store."""
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class TodoItem:
    id: int
    title: str
    completed: bool = False
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

class TodoStore:
    def __init__(self):
        self._todos: dict[int, TodoItem] = {}
        self._next_id: int = 1

    def add(self, title: str) -> TodoItem:
        if not title or not title.strip():
            raise ValueError("Title cannot be empty")
        item = TodoItem(id=self._next_id, title=title.strip())
        self._todos[self._next_id] = item
        self._next_id += 1
        return item

    def get(self, todo_id: int) -> TodoItem:
        if todo_id not in self._todos:
            raise KeyError(f"Todo {todo_id} not found")
        return self._todos[todo_id]

    def list_all(self) -> list[TodoItem]:
        return list(self._todos.values())

    def complete(self, todo_id: int) -> TodoItem:
        item = self.get(todo_id)
        item.completed = True
        return item

    def delete(self, todo_id: int) -> bool:
        if todo_id in self._todos:
            del self._todos[todo_id]
            return True
        return False
