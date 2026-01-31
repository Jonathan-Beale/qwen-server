#!/usr/bin/env python3
"""
Generate synthetic training data for tool-calling behavior

This module generates single-turn training examples with consistent XML formatting.
For multi-turn conversations and enriched data, see generate_enriched_data.py
"""

import argparse
import json
import random
import re
from pathlib import Path

# Tool definitions
TOOLS = {
    "read_file": {"path": "file path"},
    "write_file": {"path": "file path", "content": "file content"},
    "edit_file": {"path": "file path", "find": "text to find", "replace": "replacement text"},
    "bash": {"command": "shell command"},
    "list_files": {"path": "directory path", "pattern": "file pattern"},
    "search_files": {"pattern": "search pattern", "path": "directory path"},
    "summarize": {"content": "text to summarize"},
}


def validate_tool_call_xml(xml_str: str) -> bool:
    """Validate that XML uses correct <tool_call> format.

    Returns True if the XML is properly formatted, False otherwise.
    """
    # Check for required tags
    required_tags = ["<tool_call>", "</tool_call>", "<tool_name>", "</tool_name>",
                     "<parameters>", "</parameters>"]
    for tag in required_tags:
        if tag not in xml_str:
            return False

    # Check for incorrect/inconsistent formats that should be rejected
    bad_patterns = [
        r"<api_call>",
        r"<function_call",
        r"<client>",
        r"<invoke\s",
        r"<tool\s+name=",
    ]
    for pattern in bad_patterns:
        if re.search(pattern, xml_str):
            return False

    return True


def generate_tool_call_xml(tool_name, params):
    """Generate properly formatted tool call XML.

    This is the ONLY correct format for tool calls:
    <tool_call>
    <tool_name>...</tool_name>
    <parameters>
    <param>value</param>
    </parameters>
    </tool_call>
    """
    param_xml = "\n".join([f"<{k}>{v}</{k}>" for k, v in params.items()])
    xml = f"""<tool_call>
<tool_name>{tool_name}</tool_name>
<parameters>
{param_xml}
</parameters>
</tool_call>"""

    # Validate our own output
    assert validate_tool_call_xml(xml), f"Generated invalid XML for {tool_name}"
    return xml

# Training examples templates
EXAMPLES = [
    # Simple file operations
    {
        "instruction": "Write a Python script called hello.py that prints Hello World",
        "tool": "write_file",
        "params": {"path": "hello.py", "content": 'print("Hello World")'}
    },
    {
        "instruction": "Create a file named config.json with empty dict",
        "tool": "write_file",
        "params": {"path": "config.json", "content": "{}"}
    },
    {
        "instruction": "Write test.py with a function that adds two numbers",
        "tool": "write_file",
        "params": {"path": "test.py", "content": "def add(a, b):\n    return a + b"}
    },
    {
        "instruction": "Read the README.md file",
        "tool": "read_file",
        "params": {"path": "README.md"}
    },
    {
        "instruction": "Read config.yaml",
        "tool": "read_file",
        "params": {"path": "config.yaml"}
    },
    {
        "instruction": "Show me the contents of main.py",
        "tool": "read_file",
        "params": {"path": "main.py"}
    },

    # Directory operations
    {
        "instruction": "List all Python files in the src directory",
        "tool": "list_files",
        "params": {"path": "src", "pattern": "*.py"}
    },
    {
        "instruction": "Show me files in the current directory",
        "tool": "list_files",
        "params": {"path": ".", "pattern": "*"}
    },
    {
        "instruction": "List all JavaScript files",
        "tool": "list_files",
        "params": {"path": ".", "pattern": "*.js"}
    },

    # Command execution - many variations to ensure model learns bash tool
    {
        "instruction": "Run pwd to show current directory",
        "tool": "bash",
        "params": {"command": "pwd"}
    },
    {
        "instruction": "Execute ls -la",
        "tool": "bash",
        "params": {"command": "ls -la"}
    },
    {
        "instruction": "Run git status",
        "tool": "bash",
        "params": {"command": "git status"}
    },
    {
        "instruction": "Run the command git status",
        "tool": "bash",
        "params": {"command": "git status"}
    },
    {
        "instruction": "Execute git status for me",
        "tool": "bash",
        "params": {"command": "git status"}
    },
    {
        "instruction": "Can you run git status",
        "tool": "bash",
        "params": {"command": "git status"}
    },
    {
        "instruction": "git status",
        "tool": "bash",
        "params": {"command": "git status"}
    },
    {
        "instruction": "Run git diff",
        "tool": "bash",
        "params": {"command": "git diff"}
    },
    {
        "instruction": "Execute git log --oneline -5",
        "tool": "bash",
        "params": {"command": "git log --oneline -5"}
    },
    {
        "instruction": "Run npm install",
        "tool": "bash",
        "params": {"command": "npm install"}
    },
    {
        "instruction": "Execute pip install requests",
        "tool": "bash",
        "params": {"command": "pip install requests"}
    },
    {
        "instruction": "Run python main.py",
        "tool": "bash",
        "params": {"command": "python main.py"}
    },
    {
        "instruction": "Execute the tests",
        "tool": "bash",
        "params": {"command": "pytest"}
    },
    {
        "instruction": "Run pytest",
        "tool": "bash",
        "params": {"command": "pytest"}
    },
    {
        "instruction": "Run make build",
        "tool": "bash",
        "params": {"command": "make build"}
    },
    {
        "instruction": "Execute docker ps",
        "tool": "bash",
        "params": {"command": "docker ps"}
    },
    {
        "instruction": "Show current git branch",
        "tool": "bash",
        "params": {"command": "git branch --show-current"}
    },

    # Search operations
    {
        "instruction": "Search for the text TODO in all files",
        "tool": "search_files",
        "params": {"pattern": "TODO", "path": "."}
    },
    {
        "instruction": "Find all occurrences of function in Python files",
        "tool": "search_files",
        "params": {"pattern": "def ", "path": "."}
    },

    # Complex coding tasks
    {
        "instruction": "Write a Python script in utils.py that calculates factorial",
        "tool": "write_file",
        "params": {
            "path": "utils.py",
            "content": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
        }
    },
    {
        "instruction": "Create a calculator.py file with add, subtract, multiply, divide functions",
        "tool": "write_file",
        "params": {
            "path": "calculator.py",
            "content": "def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b\n\ndef multiply(a, b):\n    return a * b\n\ndef divide(a, b):\n    return a / b if b != 0 else None"
        }
    },
    {
        "instruction": "Write a script named test_api.py that makes a GET request to an API",
        "tool": "write_file",
        "params": {
            "path": "test_api.py",
            "content": "import requests\n\nresponse = requests.get('https://api.example.com/data')\nprint(response.json())"
        }
    },
]

# Variation templates for augmentation
INSTRUCTION_VARIATIONS = [
    "Write {filename} with {description}",
    "Create a file called {filename} that {description}",
    "Make a script {filename} to {description}",
    "I need {filename} with {description}",
    "Can you write {filename} that {description}",
    "Create {filename} with the following: {description}",
]

def generate_variations(base_examples):
    """Generate variations of base examples"""
    variations = []

    # File writing variations
    scripts = [
        ("server.py", "starts a Flask web server on port 5000"),
        ("data_processor.py", "reads CSV and outputs JSON"),
        ("logger.py", "implements a simple logging class"),
        ("validator.py", "validates email addresses using regex"),
        ("scraper.py", "scrapes webpage titles"),
    ]

    for filename, description in scripts:
        instruction = random.choice(INSTRUCTION_VARIATIONS).format(
            filename=filename,
            description=description
        )
        variations.append({
            "instruction": instruction,
            "tool": "write_file",
            "params": {
                "path": filename,
                "content": f"# {description}\n# Implementation here"
            }
        })

    # Read variations
    files_to_read = ["requirements.txt", "setup.py", "config.json", "data.csv", "notes.txt"]
    read_templates = ["Read {}", "Show me {}", "Display the contents of {}", "What's in {}"]

    for file in files_to_read:
        instruction = random.choice(read_templates).format(file)
        variations.append({
            "instruction": instruction,
            "tool": "read_file",
            "params": {"path": file}
        })

    # Command variations - extensive list with multiple phrasings
    commands = [
        "echo Hello",
        "date",
        "whoami",
        "python --version",
        "pip list",
        "df -h",
        "cat /etc/os-release",
        "git status",
        "git diff",
        "git log --oneline",
        "git branch",
        "git pull",
        "git push",
        "npm install",
        "npm run build",
        "npm test",
        "pip install -r requirements.txt",
        "pytest",
        "python -m pytest",
        "make",
        "make test",
        "docker ps",
        "docker-compose up",
        "ls",
        "ls -la",
        "pwd",
        "cd src",
        "mkdir new_folder",
        "rm -rf build",
        "cp file1.txt file2.txt",
        "mv old.py new.py",
        "chmod +x script.sh",
        "curl https://api.example.com",
        "wget https://example.com/file.zip",
    ]

    command_templates = [
        "Run {}",
        "Execute {}",
        "Please run {}",
        "Can you run {}",
        "Do {}",
        "Run the command {}",
        "Execute the command {}",
    ]

    for cmd in commands:
        template = random.choice(command_templates)
        variations.append({
            "instruction": template.format(cmd),
            "tool": "bash",
            "params": {"command": cmd}
        })

    return variations

def convert_to_training_format(examples):
    """Convert examples to training format"""
    training_data = []

    for ex in examples:
        output = generate_tool_call_xml(ex["tool"], ex["params"])
        training_data.append({
            "instruction": ex["instruction"],
            "output": output,
            "input": "",  # Not using input field
        })

    return training_data

def augment_examples(examples, target_count):
    """Augment examples to reach target count through variations"""
    augmented = list(examples)

    # Additional file names for variations
    filenames = [
        "app.py", "main.py", "index.js", "utils.py", "helpers.py", "api.py",
        "database.py", "models.py", "views.py", "routes.py", "auth.py",
        "config.py", "settings.py", "constants.py", "types.py", "schema.py",
        "test_main.py", "test_utils.py", "setup.py", "manage.py", "run.py",
        "client.py", "server.py", "handler.py", "middleware.py", "parser.py",
    ]

    descriptions = [
        "handles user authentication",
        "connects to the database",
        "validates input data",
        "processes API responses",
        "formats output data",
        "logs errors to file",
        "sends email notifications",
        "generates random data",
        "parses command line args",
        "reads environment variables",
        "creates a REST endpoint",
        "implements caching logic",
        "handles file uploads",
        "compresses data",
        "encrypts sensitive info",
    ]

    # Extended command list with compound commands
    commands = [
        # Basic commands
        "ls -la", "pwd", "whoami", "date", "df -h", "free -m",
        "ls", "ls -l", "ls -la src/", "ls *.py",

        # Git - simple
        "git status", "git log --oneline -5", "git branch", "git diff",
        "git pull", "git push", "git fetch", "git stash", "git checkout main",
        "git branch -a", "git log --oneline -10", "git diff --staged",
        "git stash pop", "git stash list", "git remote -v",

        # Git - compound commands (important for consistency)
        "git add . && git commit -m 'Update code'",
        "git add . && git commit -m 'Fix bug' && git push",
        "git fetch && git merge origin/main",
        "git stash && git pull && git stash pop",
        "git checkout -b feature/new-feature",
        "git add -A && git status",

        # Python
        "python --version", "pip list", "python main.py", "python -m pytest",
        "python -c \"print('hello')\"", "python setup.py install",
        "pip install requests", "pip freeze", "pip install -r requirements.txt",
        "pip install --upgrade pip", "pip show flask",
        "pytest", "pytest -v", "pytest -x", "pytest --cov",
        "pytest tests/ -v", "python -m pytest --tb=short",

        # Node/NPM
        "npm --version", "node --version", "npm install", "npm test",
        "npm run build", "npm run dev", "npm start", "npm init -y",
        "npm install express", "npm install -D jest",

        # Docker
        "docker ps", "docker images", "docker-compose up -d", "docker-compose down",
        "docker build -t myapp .", "docker run -p 8080:80 myapp",
        "docker-compose logs -f", "docker exec -it container bash",

        # System
        "cat /etc/hostname", "uname -a", "env | head", "ps aux | head",
        "top -n 1", "free -h", "df -h", "which python",

        # File operations
        "mkdir new_dir", "rm -rf build", "cp src dest", "mv old new",
        "tar -xzf archive.tar.gz", "unzip file.zip", "head -20 file.txt",
        "chmod +x script.sh", "mkdir -p src/utils", "touch newfile.py",

        # Make
        "make", "make test", "make build", "make clean", "make install",

        # Linting/formatting
        "black .", "black --check .", "flake8 src/", "mypy src/",
        "isort .", "prettier --write .",

        # Curl/wget
        "curl localhost:8000", "curl -X POST http://localhost:8000/api",
        "curl http://localhost:8000/health", "wget https://example.com/file.zip",
    ]

    command_templates = [
        "Run {}",
        "Execute {}",
        "Please run {}",
        "Can you run {}",
        "Do {}",
        "Run the command {}",
        "Execute the command: {}",
        "{}",  # Just the command itself
        "I need you to run {}",
        "Could you execute {}",
        "Run this: {}",
        "Execute this command: {}",
        "Please execute {}",
    ]

    read_files = [
        "README.md", "package.json", "requirements.txt", "Makefile",
        "Dockerfile", "docker-compose.yml", ".gitignore", ".env.example",
        "setup.cfg", "pyproject.toml", "tsconfig.json", "webpack.config.js",
    ]

    while len(augmented) < target_count:
        # Weight bash commands more heavily (40%)
        choice = random.choices([0, 1, 2, 3], weights=[25, 20, 40, 15])[0]

        if choice == 0:  # write_file
            fname = random.choice(filenames)
            desc = random.choice(descriptions)
            template = random.choice(INSTRUCTION_VARIATIONS)
            augmented.append({
                "instruction": template.format(filename=fname, description=desc),
                "tool": "write_file",
                "params": {"path": fname, "content": f"# {desc}\n\ndef main():\n    pass\n\nif __name__ == '__main__':\n    main()"}
            })
        elif choice == 1:  # read_file
            fname = random.choice(read_files + filenames)
            read_templates = ["Read {}", "Show me {}", "Display contents of {}", "What's in {}?", "Open {}", "Cat {}"]
            augmented.append({
                "instruction": random.choice(read_templates).format(fname),
                "tool": "read_file",
                "params": {"path": fname}
            })
        elif choice == 2:  # bash - most common
            cmd = random.choice(commands)
            augmented.append({
                "instruction": random.choice(command_templates).format(cmd),
                "tool": "bash",
                "params": {"command": cmd}
            })
        else:  # list_files
            patterns = ["*.py", "*.js", "*.json", "*.md", "*.txt", "*"]
            dirs = [".", "src", "lib", "tests", "docs"]
            augmented.append({
                "instruction": f"List {random.choice(patterns)} files in {random.choice(dirs)}",
                "tool": "list_files",
                "params": {"path": random.choice(dirs), "pattern": random.choice(patterns)}
            })

    return augmented[:target_count]

def main():
    parser = argparse.ArgumentParser(description="Generate training data")
    parser.add_argument('--num_examples', type=int, default=500,
                       help="Number of training examples to generate (default: 500)")
    parser.add_argument('--output', type=str, default='/training-data/tool-calling-synthetic.jsonl',
                       help="Output file path")
    parser.add_argument('--validate', action='store_true',
                       help="Validate all generated XML")
    args = parser.parse_args()

    print(f"Generating {args.num_examples} training examples...")

    # Combine base examples and variations
    all_examples = EXAMPLES + generate_variations(EXAMPLES)

    # Augment to reach target count if needed
    if args.num_examples > len(all_examples):
        all_examples = augment_examples(all_examples, args.num_examples)
    else:
        random.shuffle(all_examples)
        all_examples = all_examples[:args.num_examples]

    # Convert to training format
    training_data = convert_to_training_format(all_examples)

    # Validate XML consistency
    print("\nValidating XML format consistency...")
    valid_count = 0
    invalid_count = 0
    for item in training_data:
        output = item.get("output", "")
        if "<tool_call>" in output:
            if validate_tool_call_xml(output):
                valid_count += 1
            else:
                invalid_count += 1
                if args.validate:
                    print(f"  Invalid XML in: {item['instruction'][:50]}...")

    print(f"  Valid tool calls: {valid_count}")
    print(f"  Invalid tool calls: {invalid_count}")
    if valid_count + invalid_count > 0:
        consistency = valid_count / (valid_count + invalid_count) * 100
        print(f"  Consistency: {consistency:.1f}%")

    # Shuffle
    random.shuffle(training_data)

    # Save as JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")

    print(f"\nGenerated {len(training_data)} training examples")
    print(f"Saved to: {output_path}")

    # Count by tool type
    tool_counts = {}
    for item in training_data:
        output = item.get("output", "")
        if "<tool_name>" in output:
            # Extract tool name
            match = re.search(r"<tool_name>(\w+)</tool_name>", output)
            if match:
                tool = match.group(1)
                tool_counts[tool] = tool_counts.get(tool, 0) + 1

    print("\nTool distribution:")
    for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        print(f"  {tool}: {count}")

    # Show sample
    print("\nSample training example:")
    print(json.dumps(training_data[0], indent=2))

if __name__ == "__main__":
    main()
