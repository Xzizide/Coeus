from typing import Callable, Dict, Any, List, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import json


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable[..., Any]
    required: List[str] = field(default_factory=list)

    def to_ollama_format(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required
                }
            }
        }

    def execute(self, **kwargs) -> Any:
        return self.function(**kwargs)


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._executor = ThreadPoolExecutor(max_workers=10)

    def add_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        function: Callable[..., Any],
        required: Optional[List[str]] = None
    ) -> None:
        tool = Tool(
            name=name,
            description=description,
            parameters=parameters,
            function=function,
            required=required or []
        )
        self._tools[name] = tool

    def tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        required: Optional[List[str]] = None
    ):
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.add_tool(name, description, parameters, func, required)
            return func
        return decorator

    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def get_all_tools(self) -> List[Tool]:
        return list(self._tools.values())

    def get_ollama_tools(self) -> List[Dict[str, Any]]:
        return [tool.to_ollama_format() for tool in self._tools.values()]

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        tool = self.get_tool(name)
        if not tool:
            return {"error": f"Tool '{name}' not found"}
        try:
            result = tool.execute(**arguments)
            return result
        except Exception as e:
            return {"error": str(e)}

    def execute_tools_parallel(self, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = {}
        futures = {}

        for call in tool_calls:
            name = call["name"]
            arguments = call.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            future = self._executor.submit(self.execute_tool, name, arguments)
            futures[future] = name

        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                results[name] = {"error": str(e)}

        return results

    def has_tools(self) -> bool:
        return len(self._tools) > 0

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def remove_tool(self, name: str) -> bool:
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def clear_tools(self) -> None:
        self._tools.clear()


# Global registry for easy access
default_registry = ToolRegistry()


def add_tool(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    function: Callable[..., Any],
    required: Optional[List[str]] = None
) -> None:
    default_registry.add_tool(name, description, parameters, function, required)


def tool(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    required: Optional[List[str]] = None
):
    return default_registry.tool(name, description, parameters, required)
