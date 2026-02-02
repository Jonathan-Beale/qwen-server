import requests
import sys
import time

def stream_request(prompt, max_tokens=256):
    url = "http://localhost:8000/v1/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "tool-calling",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True
    }
    with requests.post(url, json=data, headers=headers, stream=True) as r:
        for line in r.iter_lines():
            if line:
                print(line.decode(), flush=True)

def simple_request(prompt, max_tokens=256):
    url = "http://localhost:8000/v1/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "tool-calling",
        "prompt": prompt,
        "max_tokens": max_tokens
    }
    r = requests.post(url, json=data, headers=headers)
    print(r.json())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--prompt-file', type=str, default=None)
    args = parser.parse_args()

    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            prompt = f.read()
    else:
        prompt = "def hello_world():\n    "

    print("--- Streaming Example ---")
    stream_request(prompt, max_tokens=args.max_tokens)
    print("--- Non-Streaming Example ---")
    simple_request(prompt, max_tokens=args.max_tokens)
