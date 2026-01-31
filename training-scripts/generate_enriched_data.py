#!/usr/bin/env python3
"""
Generate enriched training data for multi-stage coding agent.

Combines:
- Single-turn tool calls (with consistency fixes)
- Multi-turn conversations
- Production-quality code generation
- Error recovery scenarios
- Task decomposition examples
- Context management patterns
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any

from code_templates import CODE_TEMPLATES, generate_crud_class, generate_api_routes
from generate_multi_turn import generate_multi_turn_data, tool_call


# =============================================================================
# XML FORMAT VALIDATION
# =============================================================================

def validate_tool_call_xml(xml_str: str) -> bool:
    """Validate that XML uses correct <tool_call> format."""
    required_tags = ["<tool_call>", "</tool_call>", "<tool_name>", "</tool_name>",
                     "<parameters>", "</parameters>"]

    # Check for correct tags
    for tag in required_tags:
        if tag not in xml_str:
            return False

    # Check for incorrect formats
    bad_patterns = ["<api_call>", "<client>", "<function_call>", "<invoke>"]
    for pattern in bad_patterns:
        if pattern in xml_str:
            return False

    return True


def generate_tool_call_xml(tool_name: str, params: Dict[str, str]) -> str:
    """Generate properly formatted tool call XML."""
    param_xml = "\n".join(f"<{k}>{v}</{k}>" for k, v in params.items())
    return f"""<tool_call>
<tool_name>{tool_name}</tool_name>
<parameters>
{param_xml}
</parameters>
</tool_call>"""


# =============================================================================
# SINGLE-TURN EXAMPLES WITH CONSISTENCY FIXES
# =============================================================================

def generate_bash_examples(count: int = 500) -> List[Dict[str, Any]]:
    """Generate bash command examples with consistent XML format."""

    # Comprehensive command list
    commands = [
        # Git commands
        ("git status", "Check git status"),
        ("git diff", "Show unstaged changes"),
        ("git diff --staged", "Show staged changes"),
        ("git log --oneline -10", "Show recent commits"),
        ("git branch", "List branches"),
        ("git branch -a", "List all branches"),
        ("git checkout main", "Switch to main branch"),
        ("git checkout -b feature/new", "Create and switch to new branch"),
        ("git add .", "Stage all changes"),
        ("git add -A", "Stage all changes including deletions"),
        ("git commit -m 'Update code'", "Commit with message"),
        ("git push", "Push to remote"),
        ("git push -u origin main", "Push and set upstream"),
        ("git pull", "Pull from remote"),
        ("git fetch", "Fetch from remote"),
        ("git merge main", "Merge main into current branch"),
        ("git stash", "Stash changes"),
        ("git stash pop", "Apply stashed changes"),
        ("git reset --soft HEAD~1", "Undo last commit, keep changes"),
        ("git clean -fd", "Remove untracked files"),

        # Compound git commands
        ("git add . && git commit -m 'Fix bug'", "Stage and commit"),
        ("git add . && git commit -m 'Update' && git push", "Stage, commit, and push"),
        ("git fetch && git merge origin/main", "Fetch and merge"),
        ("git stash && git pull && git stash pop", "Stash, pull, and restore"),

        # Python commands
        ("python --version", "Check Python version"),
        ("python main.py", "Run main.py"),
        ("python -m pytest", "Run pytest"),
        ("python -m pytest -v", "Run pytest verbose"),
        ("python -m pytest test_app.py", "Run specific test file"),
        ("python -m pytest -k test_login", "Run tests matching pattern"),
        ("python -m pytest --cov=src", "Run tests with coverage"),
        ("python -c \"print('hello')\"", "Execute Python one-liner"),
        ("python setup.py install", "Install package"),
        ("python -m pip install -e .", "Install in editable mode"),

        # Pip commands
        ("pip install requests", "Install requests package"),
        ("pip install -r requirements.txt", "Install from requirements"),
        ("pip install --upgrade pip", "Upgrade pip"),
        ("pip list", "List installed packages"),
        ("pip freeze", "List packages with versions"),
        ("pip freeze > requirements.txt", "Export requirements"),
        ("pip uninstall requests -y", "Uninstall package"),
        ("pip show flask", "Show package info"),

        # NPM commands
        ("npm install", "Install dependencies"),
        ("npm install express", "Install express"),
        ("npm install -D jest", "Install jest as dev dependency"),
        ("npm run build", "Run build script"),
        ("npm run test", "Run tests"),
        ("npm run dev", "Run dev server"),
        ("npm start", "Start application"),
        ("npm init -y", "Initialize package.json"),
        ("npm update", "Update packages"),
        ("npm outdated", "Check outdated packages"),

        # Docker commands
        ("docker ps", "List running containers"),
        ("docker ps -a", "List all containers"),
        ("docker images", "List images"),
        ("docker build -t myapp .", "Build image"),
        ("docker run -p 8080:80 myapp", "Run container with port mapping"),
        ("docker-compose up", "Start services"),
        ("docker-compose up -d", "Start services detached"),
        ("docker-compose down", "Stop services"),
        ("docker-compose logs -f", "Follow logs"),
        ("docker exec -it container_name bash", "Execute bash in container"),

        # File system commands
        ("ls", "List files"),
        ("ls -la", "List files with details"),
        ("ls -la src/", "List src directory"),
        ("pwd", "Print working directory"),
        ("mkdir -p src/utils", "Create nested directories"),
        ("rm -rf build/", "Remove build directory"),
        ("cp config.example.json config.json", "Copy file"),
        ("mv old.py new.py", "Rename file"),
        ("chmod +x script.sh", "Make script executable"),
        ("cat README.md", "Display file contents"),
        ("head -20 app.py", "Show first 20 lines"),
        ("tail -f logs/app.log", "Follow log file"),
        ("wc -l src/*.py", "Count lines in Python files"),

        # System commands
        ("whoami", "Show current user"),
        ("date", "Show current date"),
        ("df -h", "Show disk usage"),
        ("free -m", "Show memory usage"),
        ("ps aux", "List processes"),
        ("top -n 1", "Show top processes once"),
        ("kill -9 1234", "Kill process"),
        ("env", "Show environment variables"),
        ("echo $PATH", "Show PATH variable"),
        ("which python", "Find python executable"),

        # Network commands
        ("curl http://localhost:8000/health", "Check health endpoint"),
        ("curl -X POST http://localhost:8000/api -d '{}'", "POST request"),
        ("wget https://example.com/file.zip", "Download file"),
        ("ping -c 3 google.com", "Ping test"),
        ("netstat -tuln", "Show listening ports"),

        # Testing commands
        ("pytest", "Run all tests"),
        ("pytest -v", "Run tests verbose"),
        ("pytest -x", "Stop on first failure"),
        ("pytest --tb=short", "Short traceback"),
        ("pytest tests/", "Run tests directory"),
        ("pytest -k 'test_user'", "Run matching tests"),
        ("pytest --cov", "Run with coverage"),
        ("pytest --cov --cov-report=html", "Coverage with HTML report"),

        # Linting/formatting
        ("black .", "Format with black"),
        ("black --check .", "Check formatting"),
        ("flake8 src/", "Run flake8"),
        ("mypy src/", "Run type checking"),
        ("isort .", "Sort imports"),
        ("prettier --write .", "Format with prettier"),
        ("eslint src/", "Run eslint"),

        # Make commands
        ("make", "Run default target"),
        ("make build", "Run build target"),
        ("make test", "Run test target"),
        ("make clean", "Run clean target"),
        ("make install", "Run install target"),
    ]

    # Instruction variations
    templates = [
        "Run {}",
        "Execute {}",
        "Please run {}",
        "Can you run {}",
        "{}",
        "Run the command: {}",
        "Execute: {}",
        "I need you to run {}",
    ]

    examples = []

    # Generate examples
    for cmd, description in commands:
        # Multiple variations per command
        for _ in range(count // len(commands) + 1):
            if len(examples) >= count:
                break

            template = random.choice(templates)
            instruction = template.format(cmd)

            # Sometimes use description-based instruction
            if random.random() < 0.3:
                instruction = description

            output = generate_tool_call_xml("bash", {"command": cmd})

            examples.append({
                "type": "single_turn",
                "category": "bash",
                "instruction": instruction,
                "output": output
            })

    random.shuffle(examples)
    return examples[:count]


def generate_file_operation_examples(count: int = 200) -> List[Dict[str, Any]]:
    """Generate file operation examples."""
    examples = []

    # Read file examples
    files_to_read = [
        "README.md", "main.py", "app.py", "config.json", "requirements.txt",
        "package.json", "setup.py", "Makefile", "Dockerfile", ".gitignore",
        "src/utils.py", "tests/test_main.py", "docs/api.md", "config.yaml"
    ]

    read_templates = [
        "Read {}",
        "Show me {}",
        "Display the contents of {}",
        "What's in {}?",
        "Open {}",
        "Let me see {}",
        "Read the file {}"
    ]

    for fname in files_to_read:
        template = random.choice(read_templates)
        examples.append({
            "type": "single_turn",
            "category": "read_file",
            "instruction": template.format(fname),
            "output": generate_tool_call_xml("read_file", {"path": fname})
        })

    # Write file examples with actual code
    write_examples = [
        ("hello.py", 'print("Hello, World!")'),
        ("config.json", '{\n    "debug": true,\n    "port": 8000\n}'),
        ("utils.py", CODE_TEMPLATES["validators"]),
        ("app.py", CODE_TEMPLATES["flask_app"]),
        ("test_app.py", CODE_TEMPLATES["pytest_basic"]),
        ("models.py", CODE_TEMPLATES["dataclass_model"]),
        ("logger.py", CODE_TEMPLATES["logger"]),
        ("requirements.txt", CODE_TEMPLATES["requirements_txt"]),
        ("Dockerfile", CODE_TEMPLATES["dockerfile"]),
        (".gitignore", CODE_TEMPLATES["gitignore"]),
    ]

    write_templates = [
        "Write {} with {}",
        "Create {} that {}",
        "Make a file {} with {}",
        "Create {} containing {}"
    ]

    descriptions = [
        "a hello world script", "basic configuration", "validation functions",
        "a Flask app", "pytest tests", "data models", "logging setup",
        "project dependencies", "Docker configuration", "git ignore rules"
    ]

    for (fname, content), desc in zip(write_examples, descriptions):
        template = random.choice(write_templates)
        examples.append({
            "type": "single_turn",
            "category": "write_file",
            "instruction": template.format(fname, desc),
            "output": generate_tool_call_xml("write_file", {"path": fname, "content": content})
        })

    # Duplicate and vary to reach count
    while len(examples) < count:
        ex = random.choice(examples).copy()
        examples.append(ex)

    random.shuffle(examples)
    return examples[:count]


def generate_correction_examples(count: int = 100) -> List[Dict[str, Any]]:
    """Generate examples that correct wrong XML formats to right format."""

    corrections = [
        {
            "wrong": "<api_call><name>bash</name><args><command>git status</command></args></api_call>",
            "right": generate_tool_call_xml("bash", {"command": "git status"}),
            "instruction": "Run git status"
        },
        {
            "wrong": "<function_call name='bash'><command>pytest</command></function_call>",
            "right": generate_tool_call_xml("bash", {"command": "pytest"}),
            "instruction": "Run pytest"
        },
        {
            "wrong": "<client><tool>read_file</tool><path>main.py</path></client>",
            "right": generate_tool_call_xml("read_file", {"path": "main.py"}),
            "instruction": "Read main.py"
        },
        {
            "wrong": "<invoke tool='bash' command='ls -la'/>",
            "right": generate_tool_call_xml("bash", {"command": "ls -la"}),
            "instruction": "List files"
        }
    ]

    examples = []

    # Expand corrections to reach count
    while len(examples) < count:
        for corr in corrections:
            if len(examples) >= count:
                break
            # Only include the correct format as training output
            examples.append({
                "type": "single_turn",
                "category": "correction",
                "instruction": corr["instruction"],
                "output": corr["right"]
            })

    return examples[:count]


# =============================================================================
# CODE GENERATION EXAMPLES
# =============================================================================

def generate_code_examples(count: int = 400) -> List[Dict[str, Any]]:
    """Generate production-quality code generation examples."""
    examples = []

    # Template-based examples
    template_instructions = [
        ("Create a Flask web application", "flask_app", "app.py"),
        ("Create a Flask app with database", "flask_app_db", "app.py"),
        ("Create a FastAPI application", "fastapi_app", "main.py"),
        ("Create a CLI tool with argparse", "cli_argparse", "cli.py"),
        ("Create a CLI tool with click", "cli_click", "cli.py"),
        ("Create pytest tests", "pytest_basic", "test_app.py"),
        ("Create API tests with pytest", "pytest_api", "test_api.py"),
        ("Create unittest tests", "unittest_basic", "test_main.py"),
        ("Create dataclass models", "dataclass_model", "models.py"),
        ("Create Pydantic models", "pydantic_model", "schemas.py"),
        ("Create a logger module", "logger", "logger.py"),
        ("Create a config loader", "config_loader", "config.py"),
        ("Create file utilities", "file_utils", "file_utils.py"),
        ("Create validation functions", "validators", "validators.py"),
        ("Create SQLAlchemy models", "sqlalchemy_models", "models.py"),
        ("Create database connection module", "database_connection", "database.py"),
        ("Create async HTTP client", "async_http_client", "http_client.py"),
        ("Create requirements.txt", "requirements_txt", "requirements.txt"),
        ("Create pyproject.toml", "pyproject_toml", "pyproject.toml"),
        ("Create Dockerfile", "dockerfile", "Dockerfile"),
        ("Create docker-compose.yml", "docker_compose", "docker-compose.yml"),
        ("Create .gitignore", "gitignore", ".gitignore"),
    ]

    for instruction, template_key, filename in template_instructions:
        content = CODE_TEMPLATES.get(template_key, "# Template not found")
        examples.append({
            "type": "single_turn",
            "category": "code_generation",
            "instruction": instruction,
            "output": generate_tool_call_xml("write_file", {
                "path": filename,
                "content": content
            })
        })

    # Dynamic code generation
    entities = [
        ("User", ["id", "name", "email", "password"]),
        ("Product", ["id", "name", "price", "quantity"]),
        ("Order", ["id", "user_id", "total", "status"]),
        ("Task", ["id", "title", "description", "completed"]),
        ("Comment", ["id", "post_id", "author", "content"]),
    ]

    for entity, fields in entities:
        # CRUD class
        examples.append({
            "type": "single_turn",
            "category": "code_generation",
            "instruction": f"Create a {entity} class with CRUD methods",
            "output": generate_tool_call_xml("write_file", {
                "path": f"{entity.lower()}.py",
                "content": generate_crud_class(entity, fields)
            })
        })

        # API routes
        examples.append({
            "type": "single_turn",
            "category": "code_generation",
            "instruction": f"Create Flask API routes for {entity}",
            "output": generate_tool_call_xml("write_file", {
                "path": f"{entity.lower()}_routes.py",
                "content": generate_api_routes(entity, fields)
            })
        })

    # Duplicate to reach count
    while len(examples) < count:
        ex = random.choice(examples).copy()
        examples.append(ex)

    random.shuffle(examples)
    return examples[:count]


# =============================================================================
# DIRECT ANSWER EXAMPLES (NO TOOL USE)
# =============================================================================

def generate_direct_answer_examples(count: int = 100) -> List[Dict[str, Any]]:
    """Generate examples where the model should answer directly without tools."""

    qa_pairs = [
        ("What is 2 + 2?", "4"),
        ("What is Python?", "Python is a high-level, interpreted programming language known for its clear syntax and readability. It's widely used for web development, data science, automation, and more."),
        ("Explain what a REST API is", "A REST API (Representational State Transfer) is an architectural style for web services that uses HTTP methods (GET, POST, PUT, DELETE) to perform operations on resources identified by URLs. It's stateless and typically returns JSON or XML data."),
        ("What's the difference between == and === in JavaScript?", "In JavaScript, `==` performs type coercion before comparison (loose equality), while `===` compares both value and type without coercion (strict equality). For example, `5 == '5'` is true but `5 === '5'` is false."),
        ("What is a decorator in Python?", "A decorator is a function that takes another function as input and extends its behavior without modifying it. It's denoted with the @ symbol above a function definition. Common uses include logging, timing, and authentication."),
        ("Explain Git branches", "Git branches are independent lines of development. The main/master branch is typically the production code, while feature branches allow developers to work on new features in isolation. Branches can be merged back together when ready."),
        ("What is Docker?", "Docker is a platform for developing, shipping, and running applications in containers. Containers package an application with all its dependencies, ensuring consistent behavior across different environments."),
        ("What is the difference between SQL and NoSQL?", "SQL databases are relational, use structured schemas, and are good for complex queries (e.g., PostgreSQL, MySQL). NoSQL databases are non-relational, offer flexible schemas, and scale horizontally (e.g., MongoDB, Redis)."),
        ("Explain async/await in JavaScript", "async/await is syntax for handling asynchronous operations. An async function returns a Promise, and await pauses execution until the Promise resolves. It makes asynchronous code look and behave more like synchronous code."),
        ("What is a virtual environment in Python?", "A virtual environment is an isolated Python environment that allows you to install packages specific to a project without affecting the global Python installation. Created with `python -m venv` or tools like conda."),
    ]

    examples = []
    for question, answer in qa_pairs:
        examples.append({
            "type": "single_turn",
            "category": "direct_answer",
            "instruction": question,
            "output": answer
        })

    # Duplicate to reach count
    while len(examples) < count:
        ex = random.choice(examples).copy()
        examples.append(ex)

    return examples[:count]


# =============================================================================
# MAIN GENERATOR
# =============================================================================

def generate_enriched_data(
    num_examples: int = 1700,
    single_turn_count: int = 500,
    multi_turn_count: int = 300,
    code_gen_count: int = 400,
    error_recovery_count: int = 200,
    task_decomp_count: int = 200,
    context_mgmt_count: int = 100
) -> List[Dict[str, Any]]:
    """Generate complete enriched training dataset."""

    all_data = []

    print("Generating training data...")

    # Phase 1: Single-turn tool calls with consistency fixes
    print(f"  - Bash commands ({single_turn_count // 2} examples)...")
    all_data.extend(generate_bash_examples(single_turn_count // 2))

    print(f"  - File operations ({single_turn_count // 4} examples)...")
    all_data.extend(generate_file_operation_examples(single_turn_count // 4))

    print(f"  - Correction examples ({single_turn_count // 4} examples)...")
    all_data.extend(generate_correction_examples(single_turn_count // 4))

    # Phase 4: Production-quality code generation
    print(f"  - Code generation ({code_gen_count} examples)...")
    all_data.extend(generate_code_examples(code_gen_count))

    # Direct answers (no tool use)
    print(f"  - Direct answer examples (100 examples)...")
    all_data.extend(generate_direct_answer_examples(100))

    # Phase 2, 3, 5, 6: Multi-turn conversations
    # These include error recovery, task decomposition, and context management
    multi_count = multi_turn_count + error_recovery_count + task_decomp_count + context_mgmt_count
    print(f"  - Multi-turn conversations ({multi_count} examples)...")
    all_data.extend(generate_multi_turn_data(multi_count))

    # Shuffle and trim
    random.shuffle(all_data)

    # Validate all tool calls
    valid_count = 0
    for item in all_data:
        if item.get("type") == "single_turn":
            output = item.get("output", "")
            if "<tool_call>" in output:
                if validate_tool_call_xml(output):
                    valid_count += 1
                else:
                    print(f"  Warning: Invalid XML in: {item.get('instruction', '')[:50]}")

    print(f"\nGenerated {len(all_data)} total examples")
    print(f"  - Single-turn: {sum(1 for x in all_data if x.get('type') == 'single_turn')}")
    print(f"  - Multi-turn: {sum(1 for x in all_data if x.get('type') == 'multi_turn')}")

    return all_data[:num_examples]


def convert_to_training_format(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert enriched data to training format."""
    training_data = []

    for item in data:
        if item.get("type") == "single_turn":
            training_data.append({
                "type": "single_turn",
                "instruction": item["instruction"],
                "output": item["output"],
                "input": ""
            })
        elif item.get("type") == "multi_turn":
            training_data.append({
                "type": "multi_turn",
                "conversation": item["conversation"]
            })

    return training_data


def main():
    parser = argparse.ArgumentParser(description="Generate enriched training data")
    parser.add_argument('--num_examples', type=int, default=1700,
                       help="Total number of examples to generate")
    parser.add_argument('--output', type=str, default='/training-data/enriched.jsonl',
                       help="Output file path")
    parser.add_argument('--single_turn', type=int, default=500,
                       help="Number of single-turn examples")
    parser.add_argument('--multi_turn', type=int, default=300,
                       help="Number of multi-turn examples")
    parser.add_argument('--code_gen', type=int, default=400,
                       help="Number of code generation examples")
    args = parser.parse_args()

    # Generate data
    data = generate_enriched_data(
        num_examples=args.num_examples,
        single_turn_count=args.single_turn,
        multi_turn_count=args.multi_turn,
        code_gen_count=args.code_gen
    )

    # Convert to training format
    training_data = convert_to_training_format(data)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")

    print(f"\nSaved to: {output_path}")

    # Show statistics
    categories = {}
    for item in data:
        cat = item.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print("\nCategory breakdown:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # Show sample
    print("\nSample single-turn example:")
    single = next((x for x in training_data if x.get("type") == "single_turn"), None)
    if single:
        print(f"  Instruction: {single['instruction'][:60]}...")
        print(f"  Output: {single['output'][:100]}...")

    print("\nSample multi-turn example:")
    multi = next((x for x in training_data if x.get("type") == "multi_turn"), None)
    if multi:
        print(f"  Turns: {len(multi['conversation'])}")
        print(f"  First turn: {multi['conversation'][0]['content'][:60]}...")


if __name__ == "__main__":
    main()
