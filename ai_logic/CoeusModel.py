import os
import json
import ollama
from dotenv import load_dotenv
from memory import ConversationMemory
from tools import ToolRegistry
from builtin_tools import register_all_builtin_tools

load_dotenv()

class Coeus:
    def __init__(self, max_history_turns: int = 10):
        self.model = os.getenv("MODEL_NAME")
        self.base_system_prompt = """You are Coeus, a memelord. You only respond with the funniest answer possible.

Use web_search for current info, then give a hilarious response based on what you found."""
        self.memory = ConversationMemory()
        self.tools = ToolRegistry()
        register_all_builtin_tools(self.tools)
        self.conversation_history = []
        self.max_history_turns = max_history_turns

    def add_tool(self, name, description, parameters, function, required=None):
        self.tools.add_tool(name, description, parameters, function, required)

    def clear_history(self):
        self.conversation_history = []

    def _add_to_history(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > self.max_history_turns * 2:
            self.conversation_history = self.conversation_history[-self.max_history_turns * 2:]

    def _build_system_prompt(self, user_message: str) -> str:
        relevant_memories = self.memory.search_memories(user_message, n_results=5)
        memory_context = self.memory.format_memories_for_prompt(relevant_memories)

        if memory_context:
            return f"{self.base_system_prompt}\n\n{memory_context}"
        return self.base_system_prompt

    def _process_tool_calls(self, tool_calls: list, messages: list) -> list:
        tool_call_list = []
        for tool_call in tool_calls:
            func = tool_call.get("function", {})
            name = func.get("name")
            args = func.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            tool_call_list.append({"name": name, "arguments": args})

        results = self.tools.execute_tools_parallel(tool_call_list)

        for tool_call in tool_calls:
            func = tool_call.get("function", {})
            name = func.get("name")
            result = results.get(name, {"error": "Tool not found"})

            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call]
            })
            messages.append({
                "role": "tool",
                "content": json.dumps(result, default=str)
            })

        return messages

    def chat(self, message: str, on_tool_call=None):
        system_prompt = self._build_system_prompt(message)

        self._add_to_history("user", message)

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation_history)

        ollama_tools = self.tools.get_ollama_tools() if self.tools.has_tools() else None
        max_iterations = 10

        for _ in range(max_iterations):
            if ollama_tools:
                response = ollama.chat(
                    model=self.model,
                    messages=messages,
                    tools=ollama_tools,
                    stream=False,
                    options={'num_ctx':4096},
                )
            else:
                stream = ollama.chat(model=self.model, messages=messages, stream=True, options={'num_ctx':4096})
                full_response = ""
                for chunk in stream:
                    content = chunk["message"]["content"]
                    full_response += content
                    yield content
                self._add_to_history("assistant", full_response)
                self.memory.add_memory(message, full_response)
                return

            msg = response.get("message", {})
            tool_calls = msg.get("tool_calls")

            if tool_calls:
                if on_tool_call:
                    for tc in tool_calls:
                        on_tool_call(tc.get("function", {}).get("name"), tc.get("function", {}).get("arguments"))

                messages = self._process_tool_calls(tool_calls, messages)
                continue

            content = msg.get("content", "")
            if content:
                yield content
                self._add_to_history("assistant", content)
                self.memory.add_memory(message, content)
            return

        yield "[Max tool iterations reached]"

    def chat_streaming(self, message: str):
        system_prompt = self._build_system_prompt(message)

        self._add_to_history("user", message)

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation_history)

        stream = ollama.chat(model=self.model, messages=messages, stream=True, options={'num_ctx':4096})
        full_response = ""
        for chunk in stream:
            content = chunk["message"]["content"]
            full_response += content
            yield content
        self._add_to_history("assistant", full_response)
        self.memory.add_memory(message, full_response)


if __name__ == "__main__":
    coeus = Coeus()

    print("Coeus initialized with tools:", coeus.tools.list_tools())
    print("Commands: /clear (long-term memory), /reset (session), /count, /tools, /notools")

    use_tools = True

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "/clear":
            count = coeus.memory.clear_memories()
            print(f"Cleared {count} long-term memories.")
            continue
        if user_input.lower() == "/reset":
            coeus.clear_history()
            print("Session history cleared.")
            continue
        if user_input.lower() == "/count":
            print(f"Long-term memories: {coeus.memory.get_memory_count()}")
            print(f"Session messages: {len(coeus.conversation_history)}")
            continue
        if user_input.lower() == "/tools":
            print(f"Available tools: {coeus.tools.list_tools()}")
            use_tools = True
            print("Tool use: enabled")
            continue
        if user_input.lower() == "/notools":
            use_tools = False
            print("Tool use: disabled")
            continue

        def tool_callback(name, args):
            print(f"\n[Using tool: {name} with {args}]")

        print("Coeus: ", end="")

        if use_tools:
            for chunk in coeus.chat(user_input, on_tool_call=tool_callback):
                print(chunk, end="", flush=True)
        else:
            for chunk in coeus.chat_streaming(user_input):
                print(chunk, end="", flush=True)
        print("")