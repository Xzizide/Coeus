import os
import math
import json
import requests
from pathlib import Path
from typing import Optional
from datetime import datetime
from tools import ToolRegistry


def register_calculator_tools(registry: ToolRegistry) -> None:
    @registry.tool(
        name="calculate",
        description="Evaluate a mathematical expression. Supports basic arithmetic, powers, roots, and common math functions.",
        parameters={
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'sin(3.14)')"
            }
        },
        required=["expression"]
    )
    def calculate(expression: str) -> dict:
        safe_dict = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "sqrt": math.sqrt,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "log": math.log, "log10": math.log10, "log2": math.log2,
            "exp": math.exp, "floor": math.floor, "ceil": math.ceil,
            "pi": math.pi, "e": math.e
        }
        try:
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return {"result": result, "expression": expression}
        except Exception as e:
            return {"error": str(e), "expression": expression}


def register_filesystem_tools(registry: ToolRegistry, base_path: Optional[str] = None) -> None:
    base = Path(base_path) if base_path else Path.cwd()

    def safe_path(path: str) -> Path:
        resolved = (base / path).resolve()
        if not str(resolved).startswith(str(base.resolve())):
            raise PermissionError("Access denied: path outside allowed directory")
        return resolved

    @registry.tool(
        name="read_file",
        description="Read the contents of a file",
        parameters={
            "path": {
                "type": "string",
                "description": "Path to the file to read (relative to base directory)"
            }
        },
        required=["path"]
    )
    def read_file(path: str) -> dict:
        try:
            file_path = safe_path(path)
            content = file_path.read_text(encoding="utf-8")
            return {"content": content, "path": str(file_path)}
        except Exception as e:
            return {"error": str(e)}

    @registry.tool(
        name="write_file",
        description="Write content to a file (creates or overwrites)",
        parameters={
            "path": {
                "type": "string",
                "description": "Path to the file to write"
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file"
            }
        },
        required=["path", "content"]
    )
    def write_file(path: str, content: str) -> dict:
        try:
            file_path = safe_path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return {"success": True, "path": str(file_path)}
        except Exception as e:
            return {"error": str(e)}

    @registry.tool(
        name="list_directory",
        description="List files and directories in a path",
        parameters={
            "path": {
                "type": "string",
                "description": "Path to the directory to list (default: current directory)"
            }
        },
        required=[]
    )
    def list_directory(path: str = ".") -> dict:
        try:
            dir_path = safe_path(path)
            items = []
            for item in dir_path.iterdir():
                items.append({
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None
                })
            return {"items": items, "path": str(dir_path)}
        except Exception as e:
            return {"error": str(e)}

    @registry.tool(
        name="file_exists",
        description="Check if a file or directory exists",
        parameters={
            "path": {
                "type": "string",
                "description": "Path to check"
            }
        },
        required=["path"]
    )
    def file_exists(path: str) -> dict:
        try:
            file_path = safe_path(path)
            exists = file_path.exists()
            return {
                "exists": exists,
                "is_file": file_path.is_file() if exists else False,
                "is_directory": file_path.is_dir() if exists else False,
                "path": str(file_path)
            }
        except Exception as e:
            return {"error": str(e)}


def register_web_tools(registry: ToolRegistry) -> None:
    def _fetch_content(url: str, max_chars: int = 5000) -> str:
        import re
        try:
            response = requests.get(url, timeout=10, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            response.raise_for_status()
            html = response.text

            # Try BeautifulSoup first for better extraction
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')

                # Remove unwanted elements
                for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'noscript', 'iframe']):
                    tag.decompose()

                # Try to find main content
                main = soup.find('main') or soup.find('article') or soup.find(class_=re.compile(r'content|article|post|entry'))
                if main:
                    text = main.get_text(separator=' ', strip=True)
                else:
                    text = soup.get_text(separator=' ', strip=True)

            except ImportError:
                # Fallback to regex
                text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<[^>]+>', ' ', text)

            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:max_chars] if text else ""
        except Exception as e:
            return f"(Failed to fetch: {str(e)[:50]})"

    @registry.tool(
        name="web_search",
        description="Search the web. Returns snippets with key info, plus fetched page content when available.",
        parameters={
            "query": {
                "type": "string",
                "description": "The search query"
            }
        },
        required=["query"]
    )
    def web_search(query: str) -> dict:
        try:
            from ddgs import DDGS
            raw_results = list(DDGS().text(query, max_results=5))

            output_parts = []
            for i, r in enumerate(raw_results, 1):
                title = r.get("title", "")
                snippet = r.get("body", "")
                url = r.get("href", "")

                part = f"[{i}] {title}\n{snippet}"

                # Try to fetch page content
                if url:
                    content = _fetch_content(url)
                    if content and len(content) > 100 and "failed" not in content.lower():
                        part += f"\n\nExtracted content: {content[:2000]}"

                output_parts.append(part)

            return {
                "query": query,
                "results": "\n\n---\n\n".join(output_parts)
            }
        except ImportError:
            return {"error": "ddgs not installed. Run: pip install ddgs"}
        except Exception as e:
            return {"error": str(e)}



def register_datetime_tools(registry: ToolRegistry) -> None:
    @registry.tool(
        name="get_current_time",
        description="Get the current date and time",
        parameters={
            "format": {
                "type": "string",
                "description": "DateTime format string (default: ISO format)"
            }
        },
        required=[]
    )
    def get_current_time(format: str = None) -> dict:
        now = datetime.now()
        if format:
            return {"time": now.strftime(format), "format": format}
        return {"time": now.isoformat(), "format": "ISO"}


def register_json_tools(registry: ToolRegistry) -> None:
    @registry.tool(
        name="parse_json",
        description="Parse a JSON string into structured data",
        parameters={
            "json_string": {
                "type": "string",
                "description": "The JSON string to parse"
            }
        },
        required=["json_string"]
    )
    def parse_json(json_string: str) -> dict:
        try:
            data = json.loads(json_string)
            return {"data": data, "success": True}
        except json.JSONDecodeError as e:
            return {"error": str(e), "success": False}


def register_all_builtin_tools(registry: ToolRegistry, filesystem_base: Optional[str] = None) -> None:
    register_calculator_tools(registry)
    register_filesystem_tools(registry, filesystem_base)
    register_web_tools(registry)
    register_datetime_tools(registry)
    register_json_tools(registry)
