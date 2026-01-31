#!/usr/bin/env python3
"""
Production-quality code templates for training data generation.

These templates provide real, working implementations instead of placeholders.
"""

# =============================================================================
# FLASK/WEB TEMPLATES
# =============================================================================

FLASK_APP_BASIC = '''from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({"message": "Hello from API"})

@app.route('/api/data', methods=['POST'])
def post_data():
    data = request.get_json()
    return jsonify({"received": data}), 201

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
'''

FLASK_APP_WITH_DB = '''from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Item(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)

    def to_dict(self):
        return {"id": self.id, "name": self.name, "description": self.description}

@app.route('/items', methods=['GET'])
def get_items():
    items = Item.query.all()
    return jsonify([item.to_dict() for item in items])

@app.route('/items', methods=['POST'])
def create_item():
    data = request.get_json()
    item = Item(name=data['name'], description=data.get('description'))
    db.session.add(item)
    db.session.commit()
    return jsonify(item.to_dict()), 201

@app.route('/items/<int:item_id>', methods=['GET'])
def get_item(item_id):
    item = Item.query.get_or_404(item_id)
    return jsonify(item.to_dict())

@app.route('/items/<int:item_id>', methods=['DELETE'])
def delete_item(item_id):
    item = Item.query.get_or_404(item_id)
    db.session.delete(item)
    db.session.commit()
    return '', 204

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True)
'''

FASTAPI_APP = '''from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(title="API Service")

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float

items_db: List[Item] = []

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/items", response_model=List[Item])
def list_items():
    return items_db

@app.post("/items", response_model=Item)
def create_item(item: Item):
    items_db.append(item)
    return item

@app.get("/items/{item_id}", response_model=Item)
def get_item(item_id: int):
    if item_id < 0 or item_id >= len(items_db):
        raise HTTPException(status_code=404, detail="Item not found")
    return items_db[item_id]
'''

# =============================================================================
# CLI TEMPLATES
# =============================================================================

CLI_ARGPARSE = '''import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Command line tool')
    parser.add_argument('input', help='Input file path')
    parser.add_argument('-o', '--output', help='Output file path', default='output.txt')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--format', choices=['json', 'csv', 'text'], default='text',
                        help='Output format')

    args = parser.parse_args()

    if args.verbose:
        print(f"Processing {args.input}...")

    # Process input file
    try:
        with open(args.input, 'r') as f:
            data = f.read()
    except FileNotFoundError:
        print(f"Error: File '{args.input}' not found", file=sys.stderr)
        sys.exit(1)

    # Write output
    with open(args.output, 'w') as f:
        f.write(data)

    if args.verbose:
        print(f"Output written to {args.output}")

if __name__ == '__main__':
    main()
'''

CLI_CLICK = '''import click

@click.group()
@click.version_option(version='1.0.0')
def cli():
    """A command line tool."""
    pass

@cli.command()
@click.argument('name')
@click.option('--greeting', '-g', default='Hello', help='Greeting to use')
def greet(name, greeting):
    """Greet someone."""
    click.echo(f"{greeting}, {name}!")

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose mode')
def process(input_file, output_file, verbose):
    """Process a file."""
    if verbose:
        click.echo(f"Processing {input_file}...")

    with open(input_file, 'r') as f:
        data = f.read()

    with open(output_file, 'w') as f:
        f.write(data.upper())

    if verbose:
        click.echo(f"Output written to {output_file}")

if __name__ == '__main__':
    cli()
'''

# =============================================================================
# TESTING TEMPLATES
# =============================================================================

PYTEST_BASIC = '''import pytest

def test_addition():
    assert 1 + 1 == 2

def test_string_operations():
    s = "hello"
    assert s.upper() == "HELLO"
    assert s.capitalize() == "Hello"
    assert len(s) == 5

class TestCalculator:
    def test_add(self):
        assert add(2, 3) == 5

    def test_subtract(self):
        assert subtract(5, 3) == 2

    def test_divide_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            divide(1, 0)

@pytest.fixture
def sample_data():
    return {"name": "test", "value": 42}

def test_with_fixture(sample_data):
    assert sample_data["name"] == "test"
    assert sample_data["value"] == 42

@pytest.mark.parametrize("input,expected", [
    (1, 1),
    (2, 4),
    (3, 9),
    (4, 16),
])
def test_square(input, expected):
    assert input ** 2 == expected
'''

PYTEST_API = '''import pytest
from fastapi.testclient import TestClient
from app import app

@pytest.fixture
def client():
    return TestClient(app)

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_create_item(client):
    item_data = {"name": "Test Item", "price": 9.99}
    response = client.post("/items", json=item_data)
    assert response.status_code == 200
    assert response.json()["name"] == "Test Item"

def test_list_items(client):
    response = client.get("/items")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_item_not_found(client):
    response = client.get("/items/999")
    assert response.status_code == 404
'''

UNITTEST_BASIC = '''import unittest

class TestStringMethods(unittest.TestCase):

    def setUp(self):
        self.test_string = "hello world"

    def tearDown(self):
        pass

    def test_upper(self):
        self.assertEqual(self.test_string.upper(), "HELLO WORLD")

    def test_split(self):
        self.assertEqual(self.test_string.split(), ["hello", "world"])

    def test_contains(self):
        self.assertIn("hello", self.test_string)
        self.assertNotIn("goodbye", self.test_string)

class TestMathOperations(unittest.TestCase):

    def test_addition(self):
        self.assertEqual(1 + 1, 2)

    def test_division(self):
        self.assertAlmostEqual(1/3, 0.333, places=3)

    def test_exception(self):
        with self.assertRaises(ZeroDivisionError):
            _ = 1 / 0

if __name__ == '__main__':
    unittest.main()
'''

# =============================================================================
# DATA CLASS / MODEL TEMPLATES
# =============================================================================

DATACLASS_MODEL = '''from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime

@dataclass
class User:
    id: int
    username: str
    email: str
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    roles: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.email or '@' not in self.email:
            raise ValueError("Invalid email address")

    def has_role(self, role: str) -> bool:
        return role in self.roles

    def add_role(self, role: str) -> None:
        if role not in self.roles:
            self.roles.append(role)

@dataclass
class Post:
    id: int
    title: str
    content: str
    author_id: int
    published: bool = False
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None

    def publish(self) -> None:
        self.published = True
        self.updated_at = datetime.now()

    def add_tag(self, tag: str) -> None:
        if tag not in self.tags:
            self.tags.append(tag)
'''

PYDANTIC_MODEL = '''from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List
from datetime import datetime

class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

    @validator('password')
    def password_strength(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain uppercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain digit')
        return v

class User(UserBase):
    id: int
    is_active: bool = True
    created_at: datetime

    class Config:
        from_attributes = True

class PostBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str
    tags: List[str] = []

class PostCreate(PostBase):
    pass

class Post(PostBase):
    id: int
    author_id: int
    published: bool = False
    created_at: datetime

    class Config:
        from_attributes = True
'''

# =============================================================================
# UTILITY TEMPLATES
# =============================================================================

LOGGER_MODULE = '''import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(
    name: str = __name__,
    level: int = logging.INFO,
    log_file: str = None
) -> logging.Logger:
    """Set up and return a configured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Default logger instance
logger = setup_logger('app', logging.INFO)
'''

CONFIG_LOADER = '''import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """Application configuration."""
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    database_url: str = "sqlite:///app.db"
    secret_key: str = "change-me-in-production"
    log_level: str = "INFO"

def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from environment and optional JSON file."""
    config = Config()

    # Load from JSON file if provided
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            data = json.load(f)
            for key, value in data.items():
                if hasattr(config, key):
                    setattr(config, key, value)

    # Override with environment variables
    config.debug = os.getenv('DEBUG', str(config.debug)).lower() == 'true'
    config.host = os.getenv('HOST', config.host)
    config.port = int(os.getenv('PORT', config.port))
    config.database_url = os.getenv('DATABASE_URL', config.database_url)
    config.secret_key = os.getenv('SECRET_KEY', config.secret_key)
    config.log_level = os.getenv('LOG_LEVEL', config.log_level)

    return config

# Global config instance
config = load_config()
'''

FILE_UTILS = '''import os
import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Union

def read_json(filepath: str) -> Dict[str, Any]:
    """Read and parse a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json(filepath: str, data: Dict[str, Any], indent: int = 2) -> None:
    """Write data to a JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def read_csv(filepath: str) -> List[Dict[str, str]]:
    """Read a CSV file and return list of dicts."""
    with open(filepath, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)

def write_csv(filepath: str, data: List[Dict[str, Any]], fieldnames: List[str] = None) -> None:
    """Write data to a CSV file."""
    if not data:
        return

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or list(data[0].keys())

    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def read_text(filepath: str) -> str:
    """Read a text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def write_text(filepath: str, content: str) -> None:
    """Write content to a text file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def ensure_dir(dirpath: str) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path = Path(dirpath)
    path.mkdir(parents=True, exist_ok=True)
    return path
'''

# =============================================================================
# DATABASE TEMPLATES
# =============================================================================

SQLALCHEMY_MODELS = '''from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    posts = relationship('Post', back_populates='author', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<User {self.username}>'

class Post(Base):
    __tablename__ = 'posts'

    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    published = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    author_id = Column(Integer, ForeignKey('users.id'), nullable=False)

    author = relationship('User', back_populates='posts')

    def __repr__(self):
        return f'<Post {self.title}>'
'''

DATABASE_CONNECTION = '''from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator
import os

DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///app.db')

engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Get a database session with automatic cleanup."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def init_db():
    """Initialize the database, creating all tables."""
    from models import Base
    Base.metadata.create_all(bind=engine)
'''

# =============================================================================
# ASYNC TEMPLATES
# =============================================================================

ASYNC_HTTP_CLIENT = '''import asyncio
import aiohttp
from typing import Dict, Any, Optional, List

class AsyncHTTPClient:
    """Async HTTP client for making API requests."""

    def __init__(self, base_url: str = "", timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()

    async def get(self, path: str, params: Dict = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        async with self._session.get(url, params=params) as response:
            response.raise_for_status()
            return await response.json()

    async def post(self, path: str, data: Dict = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        async with self._session.post(url, json=data) as response:
            response.raise_for_status()
            return await response.json()

    async def get_many(self, paths: List[str]) -> List[Dict[str, Any]]:
        """Fetch multiple URLs concurrently."""
        tasks = [self.get(path) for path in paths]
        return await asyncio.gather(*tasks)

async def main():
    async with AsyncHTTPClient("https://api.example.com") as client:
        data = await client.get("/endpoint")
        print(data)

if __name__ == "__main__":
    asyncio.run(main())
'''

# =============================================================================
# VALIDATION TEMPLATES
# =============================================================================

VALIDATORS = '''import re
from typing import Optional

def validate_email(email: str) -> bool:
    """Validate email address format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_password(password: str, min_length: int = 8) -> tuple[bool, Optional[str]]:
    """Validate password strength. Returns (is_valid, error_message)."""
    if len(password) < min_length:
        return False, f"Password must be at least {min_length} characters"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain an uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain a lowercase letter"
    if not re.search(r'\d', password):
        return False, "Password must contain a digit"
    return True, None

def validate_username(username: str) -> tuple[bool, Optional[str]]:
    """Validate username format."""
    if len(username) < 3:
        return False, "Username must be at least 3 characters"
    if len(username) > 30:
        return False, "Username must be at most 30 characters"
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return False, "Username can only contain letters, numbers, and underscores"
    return True, None

def validate_url(url: str) -> bool:
    """Validate URL format."""
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, url, re.IGNORECASE))

def sanitize_string(s: str, max_length: int = 1000) -> str:
    """Sanitize a string by removing potentially dangerous characters."""
    # Remove null bytes and control characters
    s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', s)
    # Truncate to max length
    return s[:max_length]
'''

# =============================================================================
# PROJECT STRUCTURE TEMPLATES
# =============================================================================

REQUIREMENTS_TXT = '''flask>=2.3.0
sqlalchemy>=2.0.0
pydantic>=2.0.0
python-dotenv>=1.0.0
requests>=2.31.0
pytest>=7.4.0
black>=23.0.0
'''

PYPROJECT_TOML = '''[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "myproject"
version = "0.1.0"
description = "A Python project"
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
    "flask>=2.3.0",
    "sqlalchemy>=2.0.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "black>=23.0.0",
    "mypy>=1.5.0",
]

[tool.black]
line-length = 100
target-version = ['py39']

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
'''

DOCKERFILE = '''FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
'''

DOCKER_COMPOSE = '''version: '3.8'

services:
  app:
    build: .
    ports:
      - "5000:5000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/app
      - DEBUG=false
    depends_on:
      - db

  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=app
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
'''

GITIGNORE = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
.env

# Testing
.pytest_cache/
.coverage
htmlcov/

# IDE
.vscode/
.idea/
*.swp

# Build
dist/
build/
*.egg-info/

# Database
*.db
*.sqlite3
'''

# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================

CODE_TEMPLATES = {
    # Web frameworks
    "flask_app": FLASK_APP_BASIC,
    "flask_app_db": FLASK_APP_WITH_DB,
    "fastapi_app": FASTAPI_APP,

    # CLI
    "cli_argparse": CLI_ARGPARSE,
    "cli_click": CLI_CLICK,

    # Testing
    "pytest_basic": PYTEST_BASIC,
    "pytest_api": PYTEST_API,
    "unittest_basic": UNITTEST_BASIC,

    # Models
    "dataclass_model": DATACLASS_MODEL,
    "pydantic_model": PYDANTIC_MODEL,

    # Utilities
    "logger": LOGGER_MODULE,
    "config_loader": CONFIG_LOADER,
    "file_utils": FILE_UTILS,
    "validators": VALIDATORS,

    # Database
    "sqlalchemy_models": SQLALCHEMY_MODELS,
    "database_connection": DATABASE_CONNECTION,

    # Async
    "async_http_client": ASYNC_HTTP_CLIENT,

    # Project files
    "requirements_txt": REQUIREMENTS_TXT,
    "pyproject_toml": PYPROJECT_TOML,
    "dockerfile": DOCKERFILE,
    "docker_compose": DOCKER_COMPOSE,
    "gitignore": GITIGNORE,
}


def generate_crud_class(entity_name: str, fields: list) -> str:
    """Generate a CRUD class for an entity."""
    field_params = ", ".join(fields)
    field_assignments = "\n        ".join(f"self.{f} = {f}" for f in fields)
    field_dict = ", ".join(f'"{f}": self.{f}' for f in fields)

    return f'''class {entity_name}:
    def __init__(self, {field_params}):
        {field_assignments}

    def to_dict(self):
        return {{{field_dict}}}

    @classmethod
    def from_dict(cls, data):
        return cls(**{{k: data[k] for k in [{", ".join(f'"{f}"' for f in fields)}]}})

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
'''


def generate_api_routes(entity_name: str, fields: list) -> str:
    """Generate Flask API routes for an entity."""
    entity_lower = entity_name.lower()
    return f'''from flask import Blueprint, jsonify, request

{entity_lower}_bp = Blueprint('{entity_lower}', __name__)

# In-memory storage (replace with database)
{entity_lower}s = []
next_id = 1

@{entity_lower}_bp.route('/{entity_lower}s', methods=['GET'])
def list_{entity_lower}s():
    return jsonify({entity_lower}s)

@{entity_lower}_bp.route('/{entity_lower}s', methods=['POST'])
def create_{entity_lower}():
    global next_id
    data = request.get_json()
    data['id'] = next_id
    next_id += 1
    {entity_lower}s.append(data)
    return jsonify(data), 201

@{entity_lower}_bp.route('/{entity_lower}s/<int:id>', methods=['GET'])
def get_{entity_lower}(id):
    item = next((x for x in {entity_lower}s if x['id'] == id), None)
    if not item:
        return jsonify({{"error": "Not found"}}), 404
    return jsonify(item)

@{entity_lower}_bp.route('/{entity_lower}s/<int:id>', methods=['PUT'])
def update_{entity_lower}(id):
    item = next((x for x in {entity_lower}s if x['id'] == id), None)
    if not item:
        return jsonify({{"error": "Not found"}}), 404
    data = request.get_json()
    item.update(data)
    return jsonify(item)

@{entity_lower}_bp.route('/{entity_lower}s/<int:id>', methods=['DELETE'])
def delete_{entity_lower}(id):
    global {entity_lower}s
    {entity_lower}s = [x for x in {entity_lower}s if x['id'] != id]
    return '', 204
'''


def generate_test_file(module_name: str, functions: list) -> str:
    """Generate a test file for a module."""
    imports = f"from {module_name} import {', '.join(functions)}"
    tests = "\n\n".join(f'''def test_{func}():
    # TODO: Add test implementation
    result = {func}()
    assert result is not None''' for func in functions)

    return f'''import pytest
{imports}

{tests}
'''


if __name__ == "__main__":
    # Demo: print available templates
    print("Available code templates:")
    for name in CODE_TEMPLATES:
        print(f"  - {name}")

    print("\nExample - CRUD class for 'Product':")
    print(generate_crud_class("Product", ["id", "name", "price", "quantity"]))
