#!/usr/bin/env python3
"""
Generate multi-turn conversation training data for coding agent.

This module creates realistic multi-turn conversations that teach:
- Task decomposition and planning
- Iterative development workflows
- Error recovery patterns
- Context management
"""

import random
from typing import List, Dict, Any
from code_templates import (
    CODE_TEMPLATES,
    generate_crud_class,
    generate_api_routes,
    generate_test_file,
)

# =============================================================================
# TOOL CALL GENERATION
# =============================================================================

def tool_call(tool_name: str, params: Dict[str, str]) -> str:
    """Generate consistent tool call XML."""
    param_xml = "\n".join(f"<{k}>{v}</{k}>" for k, v in params.items())
    return f"""<tool_call>
<tool_name>{tool_name}</tool_name>
<parameters>
{param_xml}
</parameters>
</tool_call>"""


# =============================================================================
# PROJECT CREATION CONVERSATIONS
# =============================================================================

def generate_flask_project_conversation() -> Dict[str, Any]:
    """Generate multi-turn conversation for creating a Flask project."""
    return {
        "type": "multi_turn",
        "category": "project_creation",
        "conversation": [
            {
                "role": "user",
                "content": "Create a Flask REST API project with user management"
            },
            {
                "role": "assistant",
                "content": """I'll create a Flask REST API project with user management. Let me break this down into steps:

1. Create the project structure with requirements.txt
2. Create the main Flask application with routes
3. Create models for users
4. Add tests for the API

Starting with the requirements file:

""" + tool_call("write_file", {
                    "path": "requirements.txt",
                    "content": "flask>=2.3.0\nflask-sqlalchemy>=3.0.0\npytest>=7.4.0\nrequests>=2.31.0"
                })
            },
            {
                "role": "tool_result",
                "content": "Created requirements.txt"
            },
            {
                "role": "assistant",
                "content": """Now creating the main Flask application with user routes:

""" + tool_call("write_file", {
                    "path": "app.py",
                    "content": CODE_TEMPLATES["flask_app_db"]
                })
            },
            {
                "role": "tool_result",
                "content": "Created app.py"
            },
            {
                "role": "assistant",
                "content": """Now I'll add tests for the API endpoints:

""" + tool_call("write_file", {
                    "path": "test_app.py",
                    "content": """import pytest
from app import app, db, Item

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
        yield client

def test_get_items_empty(client):
    response = client.get('/items')
    assert response.status_code == 200
    assert response.json == []

def test_create_item(client):
    response = client.post('/items',
        json={'name': 'Test Item', 'description': 'A test item'})
    assert response.status_code == 201
    assert response.json['name'] == 'Test Item'

def test_get_item(client):
    # First create an item
    client.post('/items', json={'name': 'Test'})
    response = client.get('/items/1')
    assert response.status_code == 200
    assert response.json['id'] == 1
"""
                })
            },
            {
                "role": "tool_result",
                "content": "Created test_app.py"
            },
            {
                "role": "assistant",
                "content": """I've created the Flask REST API project with:

1. **requirements.txt** - Dependencies (Flask, SQLAlchemy, pytest)
2. **app.py** - Main application with Item model and CRUD endpoints:
   - GET /items - List all items
   - POST /items - Create new item
   - GET /items/<id> - Get specific item
   - DELETE /items/<id> - Delete item
3. **test_app.py** - Tests for the API endpoints

To run the application:
```bash
pip install -r requirements.txt
python app.py
```

To run tests:
```bash
pytest test_app.py -v
```"""
            }
        ]
    }


def generate_cli_tool_conversation() -> Dict[str, Any]:
    """Generate multi-turn conversation for creating a CLI tool."""
    return {
        "type": "multi_turn",
        "category": "project_creation",
        "conversation": [
            {
                "role": "user",
                "content": "Build a command-line file processor that converts JSON to CSV"
            },
            {
                "role": "assistant",
                "content": """I'll create a CLI file processor for JSON to CSV conversion. Here's my plan:

1. Create the main CLI script with argparse
2. Add the conversion logic
3. Create a test file

Starting with the main script:

""" + tool_call("write_file", {
                    "path": "json2csv.py",
                    "content": """#!/usr/bin/env python3
\"\"\"Convert JSON files to CSV format.\"\"\"

import argparse
import json
import csv
import sys
from pathlib import Path

def json_to_csv(json_data, output_file, delimiter=','):
    \"\"\"Convert JSON array to CSV.\"\"\"
    if not isinstance(json_data, list):
        json_data = [json_data]

    if not json_data:
        raise ValueError("Empty JSON data")

    # Get all unique keys
    fieldnames = []
    for item in json_data:
        for key in item.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(json_data)

def main():
    parser = argparse.ArgumentParser(
        description='Convert JSON files to CSV format'
    )
    parser.add_argument('input', help='Input JSON file')
    parser.add_argument('-o', '--output', help='Output CSV file')
    parser.add_argument('-d', '--delimiter', default=',',
                        help='CSV delimiter (default: comma)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # Default output filename
    if not args.output:
        args.output = Path(args.input).stem + '.csv'

    if args.verbose:
        print(f"Reading {args.input}...")

    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{args.input}' not found", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON - {e}", file=sys.stderr)
        sys.exit(1)

    json_to_csv(data, args.output, args.delimiter)

    if args.verbose:
        print(f"Output written to {args.output}")

if __name__ == '__main__':
    main()
"""
                })
            },
            {
                "role": "tool_result",
                "content": "Created json2csv.py"
            },
            {
                "role": "assistant",
                "content": """Now adding tests for the converter:

""" + tool_call("write_file", {
                    "path": "test_json2csv.py",
                    "content": """import pytest
import json
import os
from json2csv import json_to_csv

@pytest.fixture
def sample_json(tmp_path):
    data = [
        {"name": "Alice", "age": 30, "city": "NYC"},
        {"name": "Bob", "age": 25, "city": "LA"}
    ]
    json_file = tmp_path / "test.json"
    json_file.write_text(json.dumps(data))
    return json_file

@pytest.fixture
def output_csv(tmp_path):
    return tmp_path / "output.csv"

def test_json_to_csv(sample_json, output_csv):
    with open(sample_json, 'r') as f:
        data = json.load(f)

    json_to_csv(data, str(output_csv))

    assert output_csv.exists()
    lines = output_csv.read_text().strip().split('\\n')
    assert len(lines) == 3  # header + 2 rows
    assert 'name' in lines[0]
    assert 'Alice' in lines[1]

def test_empty_json_raises():
    with pytest.raises(ValueError):
        json_to_csv([], 'output.csv')

def test_single_object():
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        json_to_csv({"key": "value"}, f.name)
        f.flush()
    os.unlink(f.name)
"""
                })
            },
            {
                "role": "tool_result",
                "content": "Created test_json2csv.py"
            },
            {
                "role": "assistant",
                "content": """I've created the CLI file processor:

1. **json2csv.py** - Main script with:
   - JSON to CSV conversion logic
   - Command-line argument parsing
   - Verbose mode and custom delimiters
   - Error handling for missing/invalid files

2. **test_json2csv.py** - Tests using pytest fixtures

Usage:
```bash
# Basic conversion
python json2csv.py data.json

# With options
python json2csv.py data.json -o output.csv -d ';' -v

# Run tests
pytest test_json2csv.py -v
```"""
            }
        ]
    }


def generate_fastapi_project_conversation() -> Dict[str, Any]:
    """Generate multi-turn conversation for creating a FastAPI project."""
    return {
        "type": "multi_turn",
        "category": "project_creation",
        "conversation": [
            {
                "role": "user",
                "content": "Create a FastAPI project with a todo list API"
            },
            {
                "role": "assistant",
                "content": """I'll create a FastAPI todo list API. Here's the plan:

1. Set up requirements with FastAPI dependencies
2. Create the main API with Pydantic models
3. Add tests using pytest and TestClient

Starting with requirements:

""" + tool_call("write_file", {
                    "path": "requirements.txt",
                    "content": "fastapi>=0.100.0\nuvicorn>=0.23.0\npydantic>=2.0.0\npytest>=7.4.0\nhttpx>=0.24.0"
                })
            },
            {
                "role": "tool_result",
                "content": "Created requirements.txt"
            },
            {
                "role": "assistant",
                "content": """Now creating the main FastAPI application:

""" + tool_call("write_file", {
                    "path": "main.py",
                    "content": """from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

app = FastAPI(title="Todo API")

class TodoCreate(BaseModel):
    title: str
    description: Optional[str] = None

class Todo(BaseModel):
    id: int
    title: str
    description: Optional[str] = None
    completed: bool = False
    created_at: datetime

# In-memory storage
todos: List[dict] = []
next_id = 1

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/todos", response_model=List[Todo])
def list_todos():
    return todos

@app.post("/todos", response_model=Todo)
def create_todo(todo: TodoCreate):
    global next_id
    new_todo = {
        "id": next_id,
        "title": todo.title,
        "description": todo.description,
        "completed": False,
        "created_at": datetime.now()
    }
    next_id += 1
    todos.append(new_todo)
    return new_todo

@app.get("/todos/{todo_id}", response_model=Todo)
def get_todo(todo_id: int):
    todo = next((t for t in todos if t["id"] == todo_id), None)
    if not todo:
        raise HTTPException(status_code=404, detail="Todo not found")
    return todo

@app.put("/todos/{todo_id}/complete")
def complete_todo(todo_id: int):
    todo = next((t for t in todos if t["id"] == todo_id), None)
    if not todo:
        raise HTTPException(status_code=404, detail="Todo not found")
    todo["completed"] = True
    return todo

@app.delete("/todos/{todo_id}")
def delete_todo(todo_id: int):
    global todos
    original_len = len(todos)
    todos = [t for t in todos if t["id"] != todo_id]
    if len(todos) == original_len:
        raise HTTPException(status_code=404, detail="Todo not found")
    return {"message": "Deleted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
                })
            },
            {
                "role": "tool_result",
                "content": "Created main.py"
            },
            {
                "role": "assistant",
                "content": """Now adding tests:

""" + tool_call("write_file", {
                    "path": "test_main.py",
                    "content": """import pytest
from fastapi.testclient import TestClient
from main import app, todos

@pytest.fixture(autouse=True)
def reset_todos():
    todos.clear()
    import main
    main.next_id = 1
    yield

@pytest.fixture
def client():
    return TestClient(app)

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_list_empty_todos(client):
    response = client.get("/todos")
    assert response.status_code == 200
    assert response.json() == []

def test_create_todo(client):
    response = client.post("/todos", json={"title": "Test task"})
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Test task"
    assert data["completed"] == False
    assert data["id"] == 1

def test_get_todo(client):
    client.post("/todos", json={"title": "Test"})
    response = client.get("/todos/1")
    assert response.status_code == 200
    assert response.json()["title"] == "Test"

def test_complete_todo(client):
    client.post("/todos", json={"title": "Test"})
    response = client.put("/todos/1/complete")
    assert response.status_code == 200
    assert response.json()["completed"] == True

def test_delete_todo(client):
    client.post("/todos", json={"title": "Test"})
    response = client.delete("/todos/1")
    assert response.status_code == 200

    response = client.get("/todos/1")
    assert response.status_code == 404

def test_todo_not_found(client):
    response = client.get("/todos/999")
    assert response.status_code == 404
"""
                })
            },
            {
                "role": "tool_result",
                "content": "Created test_main.py"
            },
            {
                "role": "assistant",
                "content": """I've created the FastAPI todo list API:

1. **requirements.txt** - Dependencies (FastAPI, uvicorn, pydantic, pytest, httpx)
2. **main.py** - API with endpoints:
   - GET /health - Health check
   - GET /todos - List all todos
   - POST /todos - Create a todo
   - GET /todos/{id} - Get specific todo
   - PUT /todos/{id}/complete - Mark as complete
   - DELETE /todos/{id} - Delete a todo
3. **test_main.py** - Comprehensive tests

To run:
```bash
pip install -r requirements.txt
python main.py
# Or: uvicorn main:app --reload

# API docs at: http://localhost:8000/docs

# Run tests:
pytest test_main.py -v
```"""
            }
        ]
    }


# =============================================================================
# BUG FIXING CONVERSATIONS
# =============================================================================

def generate_bug_fix_conversation() -> Dict[str, Any]:
    """Generate multi-turn conversation for fixing a bug."""
    return {
        "type": "multi_turn",
        "category": "bug_fixing",
        "conversation": [
            {
                "role": "user",
                "content": "The divide function in calculator.py is crashing. Can you fix it?"
            },
            {
                "role": "assistant",
                "content": """I'll investigate the divide function issue. Let me first read the file to understand the problem:

""" + tool_call("read_file", {"path": "calculator.py"})
            },
            {
                "role": "tool_result",
                "content": """def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    return a / b"""
            },
            {
                "role": "assistant",
                "content": """I found the issue. The `divide` function doesn't handle division by zero, which causes a crash when `b` is 0. Let me fix this:

""" + tool_call("edit_file", {
                    "path": "calculator.py",
                    "find": "def divide(a, b):\n    return a / b",
                    "replace": """def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b"""
                })
            },
            {
                "role": "tool_result",
                "content": "File edited successfully"
            },
            {
                "role": "assistant",
                "content": """I've fixed the divide function by adding a check for division by zero. It now raises a `ValueError` with a clear message instead of crashing with a `ZeroDivisionError`.

The function now:
- Checks if `b` is 0 before dividing
- Raises a descriptive `ValueError` if division by zero is attempted
- Performs the division normally otherwise

You can handle this in your code like:
```python
try:
    result = divide(10, 0)
except ValueError as e:
    print(f"Error: {e}")
```"""
            }
        ]
    }


def generate_test_failure_fix_conversation() -> Dict[str, Any]:
    """Generate multi-turn conversation for fixing a failing test."""
    return {
        "type": "multi_turn",
        "category": "bug_fixing",
        "conversation": [
            {
                "role": "user",
                "content": "Run the tests please"
            },
            {
                "role": "assistant",
                "content": """I'll run the tests:

""" + tool_call("bash", {"command": "pytest test_user.py -v"})
            },
            {
                "role": "tool_result",
                "content": """============================= test session starts ==============================
collected 3 items

test_user.py::test_create_user PASSED                                    [ 33%]
test_user.py::test_user_email_validation FAILED                          [ 66%]
test_user.py::test_user_str PASSED                                       [100%]

=================================== FAILURES ===================================
_____________________________ test_user_email_validation _____________________

    def test_user_email_validation():
        with pytest.raises(ValueError):
>           User("test", "invalid-email")
E           Failed: DID NOT RAISE <class 'ValueError'>

=========================== short test summary info ============================
FAILED test_user.py::test_user_email_validation - Failed: DID NOT RAISE
========================= 1 failed, 2 passed in 0.05s ========================="""
            },
            {
                "role": "assistant",
                "content": """The test `test_user_email_validation` is failing because the User class isn't validating email addresses. Let me check the User implementation:

""" + tool_call("read_file", {"path": "user.py"})
            },
            {
                "role": "tool_result",
                "content": """class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email

    def __str__(self):
        return f"User({self.name}, {self.email})"
"""
            },
            {
                "role": "assistant",
                "content": """I see the issue. The User class doesn't validate the email format. Let me add email validation:

""" + tool_call("edit_file", {
                    "path": "user.py",
                    "find": """class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email""",
                    "replace": """import re

class User:
    def __init__(self, name, email):
        self.name = name
        if not self._is_valid_email(email):
            raise ValueError(f"Invalid email address: {email}")
        self.email = email

    @staticmethod
    def _is_valid_email(email):
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))"""
                })
            },
            {
                "role": "tool_result",
                "content": "File edited successfully"
            },
            {
                "role": "assistant",
                "content": """Now let me run the tests again to verify the fix:

""" + tool_call("bash", {"command": "pytest test_user.py -v"})
            },
            {
                "role": "tool_result",
                "content": """============================= test session starts ==============================
collected 3 items

test_user.py::test_create_user PASSED                                    [ 33%]
test_user.py::test_user_email_validation PASSED                          [ 66%]
test_user.py::test_user_str PASSED                                       [100%]

========================= 3 passed in 0.05s ===================================="""
            },
            {
                "role": "assistant",
                "content": """All tests pass now. I fixed the issue by:

1. Adding email validation in the `User.__init__` method
2. Using a regex pattern to validate email format
3. Raising a `ValueError` for invalid emails

The User class now validates that emails contain:
- Characters before the @ symbol
- A domain name after @
- A valid TLD (at least 2 characters)"""
            }
        ]
    }


# =============================================================================
# ERROR RECOVERY CONVERSATIONS
# =============================================================================

def generate_import_error_recovery() -> Dict[str, Any]:
    """Generate conversation showing recovery from import error."""
    return {
        "type": "multi_turn",
        "category": "error_recovery",
        "conversation": [
            {
                "role": "user",
                "content": "Run the app"
            },
            {
                "role": "assistant",
                "content": """I'll run the application:

""" + tool_call("bash", {"command": "python app.py"})
            },
            {
                "role": "tool_result",
                "content": """Traceback (most recent call last):
  File "app.py", line 2, in <module>
    from flask import Flask
ModuleNotFoundError: No module named 'flask'"""
            },
            {
                "role": "assistant",
                "content": """The app requires Flask but it's not installed. Let me install the dependencies:

""" + tool_call("bash", {"command": "pip install flask"})
            },
            {
                "role": "tool_result",
                "content": """Collecting flask
  Using cached Flask-2.3.3-py3-none-any.whl (96 kB)
Successfully installed flask-2.3.3 werkzeug-2.3.7 jinja2-3.1.2"""
            },
            {
                "role": "assistant",
                "content": """Flask is now installed. Let me run the app again:

""" + tool_call("bash", {"command": "python app.py"})
            },
            {
                "role": "tool_result",
                "content": """* Running on http://127.0.0.1:5000
Press CTRL+C to quit"""
            },
            {
                "role": "assistant",
                "content": """The application is now running successfully at http://127.0.0.1:5000

I resolved the import error by installing the missing Flask package. If you have a `requirements.txt` file, you can install all dependencies at once with:
```bash
pip install -r requirements.txt
```"""
            }
        ]
    }


def generate_syntax_error_recovery() -> Dict[str, Any]:
    """Generate conversation showing recovery from syntax error."""
    return {
        "type": "multi_turn",
        "category": "error_recovery",
        "conversation": [
            {
                "role": "user",
                "content": "Run pytest"
            },
            {
                "role": "assistant",
                "content": """Running the tests:

""" + tool_call("bash", {"command": "pytest"})
            },
            {
                "role": "tool_result",
                "content": """============================= ERRORS ==============================
________________ ERROR collecting test_utils.py _________________
  File "test_utils.py", line 8
    def test_parse_data()
                        ^
SyntaxError: expected ':'

========================= short test summary info =========================
ERROR test_utils.py
!!!!!!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error !!!!!!!!!!!!!!!!!!!!!!!!!!!"""
            },
            {
                "role": "assistant",
                "content": """There's a syntax error in test_utils.py on line 8 - a missing colon after the function definition. Let me read the file and fix it:

""" + tool_call("read_file", {"path": "test_utils.py"})
            },
            {
                "role": "tool_result",
                "content": """import pytest
from utils import parse_data

def test_parse_empty():
    assert parse_data("") == []

def test_parse_data()
    assert parse_data("a,b,c") == ["a", "b", "c"]

def test_parse_with_spaces():
    assert parse_data("a, b, c") == ["a", "b", "c"]
"""
            },
            {
                "role": "assistant",
                "content": """I see the issue - line 8 is missing a colon. Let me fix it:

""" + tool_call("edit_file", {
                    "path": "test_utils.py",
                    "find": "def test_parse_data()",
                    "replace": "def test_parse_data():"
                })
            },
            {
                "role": "tool_result",
                "content": "File edited successfully"
            },
            {
                "role": "assistant",
                "content": """Now running the tests again:

""" + tool_call("bash", {"command": "pytest"})
            },
            {
                "role": "tool_result",
                "content": """============================= test session starts ==============================
collected 3 items

test_utils.py::test_parse_empty PASSED                                   [ 33%]
test_utils.py::test_parse_data PASSED                                    [ 66%]
test_utils.py::test_parse_with_spaces PASSED                             [100%]

========================= 3 passed in 0.02s ===================================="""
            },
            {
                "role": "assistant",
                "content": """Fixed the syntax error and all tests pass now. The issue was a missing colon (`:`) at the end of the function definition on line 8."""
            }
        ]
    }


def generate_file_not_found_recovery() -> Dict[str, Any]:
    """Generate conversation showing recovery from file not found."""
    return {
        "type": "multi_turn",
        "category": "error_recovery",
        "conversation": [
            {
                "role": "user",
                "content": "Read the config file and show me the database settings"
            },
            {
                "role": "assistant",
                "content": """I'll read the config file:

""" + tool_call("read_file", {"path": "config.json"})
            },
            {
                "role": "tool_result",
                "content": "Error: File 'config.json' not found"
            },
            {
                "role": "assistant",
                "content": """The config file doesn't exist. Let me check what configuration files are available:

""" + tool_call("list_files", {"path": ".", "pattern": "*.json"})
            },
            {
                "role": "tool_result",
                "content": """settings.json
package.json"""
            },
            {
                "role": "assistant",
                "content": """Found `settings.json`. Let me check if that contains the database configuration:

""" + tool_call("read_file", {"path": "settings.json"})
            },
            {
                "role": "tool_result",
                "content": """{
    "database": {
        "host": "localhost",
        "port": 5432,
        "name": "myapp",
        "user": "admin"
    },
    "debug": false
}"""
            },
            {
                "role": "assistant",
                "content": """Found the database settings in `settings.json`:

- **Host**: localhost
- **Port**: 5432
- **Database name**: myapp
- **User**: admin

Note: The configuration file is named `settings.json`, not `config.json`."""
            }
        ]
    }


# =============================================================================
# REFACTORING CONVERSATIONS
# =============================================================================

def generate_refactoring_conversation() -> Dict[str, Any]:
    """Generate multi-turn conversation for refactoring code."""
    return {
        "type": "multi_turn",
        "category": "refactoring",
        "conversation": [
            {
                "role": "user",
                "content": "Refactor utils.py to split it into separate modules"
            },
            {
                "role": "assistant",
                "content": """I'll analyze utils.py first to understand what functions it contains:

""" + tool_call("read_file", {"path": "utils.py"})
            },
            {
                "role": "tool_result",
                "content": """import json
import csv
import os
import re

def read_json(path):
    with open(path) as f:
        return json.load(f)

def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def write_csv(path, data, fieldnames):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_url(url):
    pattern = r'^https?://[^\\s/$.?#].[^\\s]*$'
    return bool(re.match(pattern, url))

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
"""
            },
            {
                "role": "assistant",
                "content": """I'll split this into logical modules:
1. `file_utils.py` - JSON, CSV, and directory operations
2. `validators.py` - Email and URL validation

Creating file_utils.py:

""" + tool_call("write_file", {
                    "path": "file_utils.py",
                    "content": """\"\"\"File utility functions for JSON, CSV, and directory operations.\"\"\"
import json
import csv
import os

def read_json(path):
    \"\"\"Read and parse a JSON file.\"\"\"
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def write_json(path, data, indent=2):
    \"\"\"Write data to a JSON file.\"\"\"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)

def read_csv(path):
    \"\"\"Read a CSV file and return list of dicts.\"\"\"
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))

def write_csv(path, data, fieldnames=None):
    \"\"\"Write data to a CSV file.\"\"\"
    if not data:
        return
    fieldnames = fieldnames or list(data[0].keys())
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def ensure_dir(path):
    \"\"\"Ensure a directory exists.\"\"\"
    os.makedirs(path, exist_ok=True)
"""
                })
            },
            {
                "role": "tool_result",
                "content": "Created file_utils.py"
            },
            {
                "role": "assistant",
                "content": """Now creating validators.py:

""" + tool_call("write_file", {
                    "path": "validators.py",
                    "content": """\"\"\"Validation utility functions.\"\"\"
import re

def validate_email(email):
    \"\"\"Validate email address format.\"\"\"
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_url(url):
    \"\"\"Validate URL format.\"\"\"
    pattern = r'^https?://[^\\s/$.?#].[^\\s]*$'
    return bool(re.match(pattern, url))
"""
                })
            },
            {
                "role": "tool_result",
                "content": "Created validators.py"
            },
            {
                "role": "assistant",
                "content": """Now updating utils.py to re-export everything for backward compatibility:

""" + tool_call("write_file", {
                    "path": "utils.py",
                    "content": """\"\"\"Utility functions - re-exports from submodules for backward compatibility.\"\"\"

from file_utils import read_json, write_json, read_csv, write_csv, ensure_dir
from validators import validate_email, validate_url

__all__ = [
    'read_json', 'write_json', 'read_csv', 'write_csv', 'ensure_dir',
    'validate_email', 'validate_url'
]
"""
                })
            },
            {
                "role": "tool_result",
                "content": "Created utils.py"
            },
            {
                "role": "assistant",
                "content": """I've refactored utils.py into separate modules:

1. **file_utils.py** - File operations (JSON, CSV, directories)
2. **validators.py** - Validation functions (email, URL)
3. **utils.py** - Re-exports everything for backward compatibility

Existing code using `from utils import read_json` will continue to work. New code can import directly:
```python
from file_utils import read_json
from validators import validate_email
```"""
            }
        ]
    }


# =============================================================================
# CONTEXT MANAGEMENT CONVERSATIONS
# =============================================================================

def generate_context_summary_conversation() -> Dict[str, Any]:
    """Generate conversation demonstrating context summarization."""
    return {
        "type": "multi_turn",
        "category": "context_management",
        "conversation": [
            {
                "role": "user",
                "content": "Let's continue building the API. Add user authentication to the Flask app we created."
            },
            {
                "role": "assistant",
                "content": """To maintain context efficiently, let me summarize what we have and then proceed with authentication.

""" + tool_call("summarize", {
                    "content": "We have a Flask app (app.py) with Item model and CRUD endpoints (/items). Using SQLAlchemy with SQLite database. Tests in test_app.py."
                })
            },
            {
                "role": "tool_result",
                "content": "Summary stored: Flask app with Item CRUD endpoints, SQLAlchemy/SQLite, tests available"
            },
            {
                "role": "assistant",
                "content": """Based on our existing Flask app, I'll add user authentication. Let me first read the current app.py to see the structure:

""" + tool_call("read_file", {"path": "app.py"})
            },
            {
                "role": "tool_result",
                "content": CODE_TEMPLATES["flask_app_db"]
            },
            {
                "role": "assistant",
                "content": """I'll add a User model and authentication endpoints. Let me update the app:

""" + tool_call("edit_file", {
                    "path": "app.py",
                    "find": "class Item(db.Model):",
                    "replace": """from werkzeug.security import generate_password_hash, check_password_hash
import secrets

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    api_key = db.Column(db.String(64), unique=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def generate_api_key(self):
        self.api_key = secrets.token_hex(32)
        return self.api_key

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if User.query.filter_by(username=data['username']).first():
        return jsonify({"error": "Username already exists"}), 400
    user = User(username=data['username'])
    user.set_password(data['password'])
    user.generate_api_key()
    db.session.add(user)
    db.session.commit()
    return jsonify({"username": user.username, "api_key": user.api_key}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data['username']).first()
    if not user or not user.check_password(data['password']):
        return jsonify({"error": "Invalid credentials"}), 401
    return jsonify({"username": user.username, "api_key": user.api_key})

class Item(db.Model):"""
                })
            },
            {
                "role": "tool_result",
                "content": "File edited successfully"
            },
            {
                "role": "assistant",
                "content": """I've added user authentication to the Flask app:

**New Model:**
- `User` with username, password_hash, and api_key

**New Endpoints:**
- `POST /register` - Create new user account
- `POST /login` - Authenticate and get API key

**Features:**
- Password hashing with werkzeug
- API key generation for stateless auth
- Duplicate username prevention

Usage:
```bash
# Register
curl -X POST http://localhost:5000/register \\
  -H "Content-Type: application/json" \\
  -d '{"username": "alice", "password": "secret123"}'

# Login
curl -X POST http://localhost:5000/login \\
  -H "Content-Type: application/json" \\
  -d '{"username": "alice", "password": "secret123"}'
```"""
            }
        ]
    }


def generate_incremental_building_conversation() -> Dict[str, Any]:
    """Generate conversation demonstrating incremental file building."""
    return {
        "type": "multi_turn",
        "category": "context_management",
        "conversation": [
            {
                "role": "user",
                "content": "Create a Python package for data validation"
            },
            {
                "role": "assistant",
                "content": """I'll create the data validation package incrementally, one file at a time.

First, let me set up the package structure with __init__.py:

""" + tool_call("write_file", {
                    "path": "validation/__init__.py",
                    "content": """\"\"\"Data validation package.\"\"\"

from .validators import (
    validate_email,
    validate_url,
    validate_phone,
    ValidationError
)

__version__ = "0.1.0"
__all__ = ['validate_email', 'validate_url', 'validate_phone', 'ValidationError']
"""
                })
            },
            {
                "role": "tool_result",
                "content": "Created validation/__init__.py"
            },
            {
                "role": "assistant",
                "content": """Now creating the core validators module:

""" + tool_call("write_file", {
                    "path": "validation/validators.py",
                    "content": """\"\"\"Core validation functions.\"\"\"
import re

class ValidationError(Exception):
    \"\"\"Raised when validation fails.\"\"\"
    pass

def validate_email(email: str) -> bool:
    \"\"\"Validate email format.\"\"\"
    if not email:
        raise ValidationError("Email cannot be empty")
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        raise ValidationError(f"Invalid email format: {email}")
    return True

def validate_url(url: str) -> bool:
    \"\"\"Validate URL format.\"\"\"
    if not url:
        raise ValidationError("URL cannot be empty")
    pattern = r'^https?://[^\\s/$.?#].[^\\s]*$'
    if not re.match(pattern, url):
        raise ValidationError(f"Invalid URL format: {url}")
    return True

def validate_phone(phone: str, country: str = "US") -> bool:
    \"\"\"Validate phone number format.\"\"\"
    if not phone:
        raise ValidationError("Phone cannot be empty")
    # Remove common separators
    digits = re.sub(r'[\\s\\-\\(\\)\\.]', '', phone)
    if country == "US":
        if not re.match(r'^\\+?1?\\d{10}$', digits):
            raise ValidationError(f"Invalid US phone format: {phone}")
    return True
"""
                })
            },
            {
                "role": "tool_result",
                "content": "Created validation/validators.py"
            },
            {
                "role": "assistant",
                "content": """Now adding tests for the validation package:

""" + tool_call("write_file", {
                    "path": "validation/tests/test_validators.py",
                    "content": """\"\"\"Tests for validators module.\"\"\"
import pytest
from validation import validate_email, validate_url, validate_phone, ValidationError

class TestEmailValidation:
    def test_valid_email(self):
        assert validate_email("user@example.com") == True
        assert validate_email("name.surname@domain.org") == True

    def test_invalid_email(self):
        with pytest.raises(ValidationError):
            validate_email("invalid-email")

    def test_empty_email(self):
        with pytest.raises(ValidationError):
            validate_email("")

class TestURLValidation:
    def test_valid_url(self):
        assert validate_url("https://example.com") == True
        assert validate_url("http://test.org/path") == True

    def test_invalid_url(self):
        with pytest.raises(ValidationError):
            validate_url("not-a-url")

class TestPhoneValidation:
    def test_valid_us_phone(self):
        assert validate_phone("555-123-4567") == True
        assert validate_phone("(555) 123-4567") == True
        assert validate_phone("+15551234567") == True

    def test_invalid_phone(self):
        with pytest.raises(ValidationError):
            validate_phone("123")
"""
                })
            },
            {
                "role": "tool_result",
                "content": "Created validation/tests/test_validators.py"
            },
            {
                "role": "assistant",
                "content": """I've created the data validation package with:

**Structure:**
```
validation/
├── __init__.py       # Package exports
├── validators.py     # Core validation functions
└── tests/
    └── test_validators.py
```

**Features:**
- `ValidationError` exception class
- `validate_email()` - Email format validation
- `validate_url()` - URL format validation
- `validate_phone()` - Phone number validation (US format)

**Usage:**
```python
from validation import validate_email, ValidationError

try:
    validate_email("user@example.com")  # Returns True
    validate_email("bad-email")  # Raises ValidationError
except ValidationError as e:
    print(f"Validation failed: {e}")
```

Run tests with: `pytest validation/tests/ -v`"""
            }
        ]
    }


# =============================================================================
# TASK DECOMPOSITION CONVERSATIONS
# =============================================================================

def generate_task_decomposition_conversation() -> Dict[str, Any]:
    """Generate conversation demonstrating task decomposition."""
    return {
        "type": "multi_turn",
        "category": "task_decomposition",
        "conversation": [
            {
                "role": "user",
                "content": "Build a simple blog API with posts and comments"
            },
            {
                "role": "assistant",
                "content": """I'll build this blog API step by step. Let me break down the task:

1. **Create project structure** - requirements.txt and main app file
2. **Define data models** - Post and Comment classes
3. **Implement API endpoints** - CRUD for posts and comments
4. **Add tests** - Test the API endpoints

Starting with step 1 - creating requirements.txt:

""" + tool_call("write_file", {
                    "path": "requirements.txt",
                    "content": "fastapi>=0.100.0\nuvicorn>=0.23.0\npydantic>=2.0.0\npytest>=7.4.0\nhttpx>=0.24.0"
                })
            },
            {
                "role": "tool_result",
                "content": "Created requirements.txt"
            },
            {
                "role": "assistant",
                "content": """Step 2 - Creating the models file with Post and Comment:

""" + tool_call("write_file", {
                    "path": "models.py",
                    "content": """from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class CommentCreate(BaseModel):
    author: str
    content: str

class Comment(CommentCreate):
    id: int
    post_id: int
    created_at: datetime

class PostCreate(BaseModel):
    title: str
    content: str
    author: str

class Post(PostCreate):
    id: int
    created_at: datetime
    comments: List[Comment] = []
"""
                })
            },
            {
                "role": "tool_result",
                "content": "Created models.py"
            },
            {
                "role": "assistant",
                "content": """Step 3 - Creating the main API with all endpoints:

""" + tool_call("write_file", {
                    "path": "main.py",
                    "content": """from fastapi import FastAPI, HTTPException
from models import Post, PostCreate, Comment, CommentCreate
from typing import List
from datetime import datetime

app = FastAPI(title="Blog API")

# Storage
posts: List[dict] = []
comments: List[dict] = []
post_id_counter = 1
comment_id_counter = 1

@app.get("/posts", response_model=List[Post])
def list_posts():
    result = []
    for post in posts:
        post_comments = [c for c in comments if c["post_id"] == post["id"]]
        result.append({**post, "comments": post_comments})
    return result

@app.post("/posts", response_model=Post)
def create_post(post: PostCreate):
    global post_id_counter
    new_post = {
        "id": post_id_counter,
        "title": post.title,
        "content": post.content,
        "author": post.author,
        "created_at": datetime.now()
    }
    post_id_counter += 1
    posts.append(new_post)
    return {**new_post, "comments": []}

@app.get("/posts/{post_id}", response_model=Post)
def get_post(post_id: int):
    post = next((p for p in posts if p["id"] == post_id), None)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    post_comments = [c for c in comments if c["post_id"] == post_id]
    return {**post, "comments": post_comments}

@app.delete("/posts/{post_id}")
def delete_post(post_id: int):
    global posts, comments
    post = next((p for p in posts if p["id"] == post_id), None)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    posts = [p for p in posts if p["id"] != post_id]
    comments = [c for c in comments if c["post_id"] != post_id]
    return {"message": "Deleted"}

@app.post("/posts/{post_id}/comments", response_model=Comment)
def add_comment(post_id: int, comment: CommentCreate):
    global comment_id_counter
    post = next((p for p in posts if p["id"] == post_id), None)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    new_comment = {
        "id": comment_id_counter,
        "post_id": post_id,
        "author": comment.author,
        "content": comment.content,
        "created_at": datetime.now()
    }
    comment_id_counter += 1
    comments.append(new_comment)
    return new_comment

@app.get("/posts/{post_id}/comments", response_model=List[Comment])
def list_comments(post_id: int):
    post = next((p for p in posts if p["id"] == post_id), None)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return [c for c in comments if c["post_id"] == post_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
                })
            },
            {
                "role": "tool_result",
                "content": "Created main.py"
            },
            {
                "role": "assistant",
                "content": """Step 4 - Adding tests for the API:

""" + tool_call("write_file", {
                    "path": "test_main.py",
                    "content": """import pytest
from fastapi.testclient import TestClient
from main import app, posts, comments

@pytest.fixture(autouse=True)
def reset_data():
    posts.clear()
    comments.clear()
    import main
    main.post_id_counter = 1
    main.comment_id_counter = 1

@pytest.fixture
def client():
    return TestClient(app)

def test_create_post(client):
    response = client.post("/posts", json={
        "title": "Test Post",
        "content": "Hello world",
        "author": "Alice"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Test Post"
    assert data["id"] == 1

def test_list_posts(client):
    client.post("/posts", json={"title": "Post 1", "content": "Content", "author": "Alice"})
    client.post("/posts", json={"title": "Post 2", "content": "Content", "author": "Bob"})

    response = client.get("/posts")
    assert response.status_code == 200
    assert len(response.json()) == 2

def test_get_post(client):
    client.post("/posts", json={"title": "Test", "content": "Content", "author": "Alice"})
    response = client.get("/posts/1")
    assert response.status_code == 200
    assert response.json()["title"] == "Test"

def test_add_comment(client):
    client.post("/posts", json={"title": "Test", "content": "Content", "author": "Alice"})
    response = client.post("/posts/1/comments", json={
        "author": "Bob",
        "content": "Great post!"
    })
    assert response.status_code == 200
    assert response.json()["content"] == "Great post!"

def test_list_comments(client):
    client.post("/posts", json={"title": "Test", "content": "Content", "author": "Alice"})
    client.post("/posts/1/comments", json={"author": "Bob", "content": "Comment 1"})
    client.post("/posts/1/comments", json={"author": "Carol", "content": "Comment 2"})

    response = client.get("/posts/1/comments")
    assert response.status_code == 200
    assert len(response.json()) == 2

def test_delete_post_removes_comments(client):
    client.post("/posts", json={"title": "Test", "content": "Content", "author": "Alice"})
    client.post("/posts/1/comments", json={"author": "Bob", "content": "Comment"})

    client.delete("/posts/1")

    response = client.get("/posts/1")
    assert response.status_code == 404
"""
                })
            },
            {
                "role": "tool_result",
                "content": "Created test_main.py"
            },
            {
                "role": "assistant",
                "content": """I've completed all 4 steps. Here's the blog API:

**Files Created:**
1. `requirements.txt` - Dependencies
2. `models.py` - Pydantic models for Post and Comment
3. `main.py` - FastAPI application with endpoints
4. `test_main.py` - Comprehensive tests

**API Endpoints:**
| Method | Path | Description |
|--------|------|-------------|
| GET | /posts | List all posts |
| POST | /posts | Create a post |
| GET | /posts/{id} | Get a post with comments |
| DELETE | /posts/{id} | Delete post and its comments |
| POST | /posts/{id}/comments | Add comment to post |
| GET | /posts/{id}/comments | List post comments |

**Run the API:**
```bash
pip install -r requirements.txt
uvicorn main:app --reload
# Docs at http://localhost:8000/docs
```

**Run tests:**
```bash
pytest test_main.py -v
```"""
            }
        ]
    }


# =============================================================================
# COMPOUND COMMAND CONVERSATIONS
# =============================================================================

def generate_compound_commands_conversation() -> Dict[str, Any]:
    """Generate conversation with compound bash commands."""
    return {
        "type": "multi_turn",
        "category": "compound_commands",
        "conversation": [
            {
                "role": "user",
                "content": "Set up a new git repository and make an initial commit"
            },
            {
                "role": "assistant",
                "content": """I'll initialize a git repository and create an initial commit:

""" + tool_call("bash", {"command": "git init && git add . && git commit -m 'Initial commit'"})
            },
            {
                "role": "tool_result",
                "content": """Initialized empty Git repository in /project/.git/
[main (root-commit) a1b2c3d] Initial commit
 5 files changed, 150 insertions(+)
 create mode 100644 app.py
 create mode 100644 requirements.txt
 create mode 100644 test_app.py"""
            },
            {
                "role": "assistant",
                "content": """Git repository initialized with all files committed. The initial commit includes 5 files with 150 lines of code."""
            }
        ]
    }


def generate_git_workflow_conversation() -> Dict[str, Any]:
    """Generate conversation with git workflow."""
    return {
        "type": "multi_turn",
        "category": "compound_commands",
        "conversation": [
            {
                "role": "user",
                "content": "Create a feature branch and push my changes"
            },
            {
                "role": "assistant",
                "content": """I'll create a feature branch, commit changes, and push to remote:

""" + tool_call("bash", {"command": "git checkout -b feature/new-feature && git add . && git commit -m 'Add new feature' && git push -u origin feature/new-feature"})
            },
            {
                "role": "tool_result",
                "content": """Switched to a new branch 'feature/new-feature'
[feature/new-feature a2b3c4d] Add new feature
 2 files changed, 45 insertions(+), 3 deletions(-)
Branch 'feature/new-feature' set up to track remote branch 'feature/new-feature' from 'origin'.
To github.com:user/repo.git
 * [new branch]      feature/new-feature -> feature/new-feature"""
            },
            {
                "role": "assistant",
                "content": """Done! Created and pushed the feature branch:
- Created branch: `feature/new-feature`
- Committed 2 changed files
- Pushed to origin with upstream tracking set"""
            }
        ]
    }


# =============================================================================
# MAIN GENERATOR
# =============================================================================

def get_all_conversation_generators():
    """Return all conversation generator functions."""
    return [
        # Project creation
        generate_flask_project_conversation,
        generate_cli_tool_conversation,
        generate_fastapi_project_conversation,

        # Bug fixing
        generate_bug_fix_conversation,
        generate_test_failure_fix_conversation,

        # Error recovery
        generate_import_error_recovery,
        generate_syntax_error_recovery,
        generate_file_not_found_recovery,

        # Refactoring
        generate_refactoring_conversation,

        # Context management
        generate_context_summary_conversation,
        generate_incremental_building_conversation,

        # Task decomposition
        generate_task_decomposition_conversation,

        # Compound commands
        generate_compound_commands_conversation,
        generate_git_workflow_conversation,
    ]


def generate_multi_turn_data(num_examples: int = 300) -> List[Dict[str, Any]]:
    """Generate multi-turn conversation training data."""
    generators = get_all_conversation_generators()
    conversations = []

    # Generate base conversations
    for gen in generators:
        conversations.append(gen())

    # Duplicate and vary to reach target
    while len(conversations) < num_examples:
        gen = random.choice(generators)
        conversations.append(gen())

    random.shuffle(conversations)
    return conversations[:num_examples]


if __name__ == "__main__":
    import json

    # Generate sample data
    data = generate_multi_turn_data(50)

    print(f"Generated {len(data)} multi-turn conversations")
    print("\nCategories:")
    categories = {}
    for conv in data:
        cat = conv.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    # Show sample
    print("\nSample conversation:")
    sample = data[0]
    print(f"Type: {sample['type']}")
    print(f"Category: {sample['category']}")
    print(f"Turns: {len(sample['conversation'])}")
    for turn in sample['conversation'][:3]:
        print(f"\n[{turn['role']}]:")
        content = turn['content'][:200] + "..." if len(turn['content']) > 200 else turn['content']
        print(content)
