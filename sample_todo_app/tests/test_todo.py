"""Tests for sample todo app."""
import pytest
from sample_todo_app.service import TodoService
from sample_todo_app.api import handle_summary, handle_create

def test_todo_creation():
    item = handle_create("Buy groceries")
    assert item.title == "Buy groceries"
    assert not item.completed

def test_empty_summary_raises_on_zero():
    """Fails currently because total is 0 and causes ZeroDivisionError."""
    empty_service = TodoService()
    summary = empty_service.get_summary()
    assert summary["total"] == 0
    assert summary["completion_rate"] == 0
