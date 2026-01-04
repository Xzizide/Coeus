import os
import json
import ollama
from dotenv import load_dotenv
from memory import ConversationMemory
from tools import ToolRegistry
from builtin_tools import register_all_builtin_tools
from rag import DocumentRAG

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
        self.rag = DocumentRAG()

    def add_tool(self, name, description, parameters, function, required=None):
        self.tools.add_tool(name, description, parameters, function, required)

    def clear_history(self):
        self.conversation_history = []

    def _add_to_history(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > self.max_history_turns * 2:
            self.conversation_history = self.conversation_history[-self.max_history_turns * 2:]

    def _build_system_prompt(self, user_message: str) -> str:
        prompt = self.base_system_prompt

        # Add RAG context if documents exist
        if self.rag.get_chunk_count() > 0:
            relevant_docs = self.rag.search_documents(user_message, n_results=5)
            rag_context = self.rag.format_context_for_prompt(relevant_docs)
            if rag_context:
                prompt += f"\n\n{rag_context}"

        # Add conversation memory context
        relevant_memories = self.memory.search_memories(user_message, n_results=5)
        memory_context = self.memory.format_memories_for_prompt(relevant_memories)
        if memory_context:
            prompt += f"\n\n{memory_context}"

        return prompt

    # RAG helper methods
    def load_documents(self):
        return self.rag.load_documents()

    def add_document(self, file_path: str):
        return self.rag.add_document(file_path)

    def search_documents(self, query: str, n_results: int = 5):
        return self.rag.search_documents(query, n_results)

    def list_documents(self):
        return self.rag.list_documents()

    def clear_rag_database(self):
        return self.rag.clear_rag_database()

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
    print("Commands: /clear, /reset, /count, /tools, /notools")
    print("RAG: /load, /docs, /cleardocs, /add <path>")

    # Auto-load documents on startup
    result = coeus.load_documents()
    if result.get("loaded"):
        print(f"Loaded {len(result['loaded'])} documents ({result['total_chunks']} chunks)")

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
            print(f"RAG chunks: {coeus.rag.get_chunk_count()}")
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
        if user_input.lower() == "/load":
            result = coeus.load_documents()
            print(f"Loaded: {result.get('loaded', [])}")
            print(f"Skipped (already loaded): {result.get('skipped', [])}")
            print(f"Total chunks: {result.get('total_chunks', 0)}")
            continue
        if user_input.lower() == "/docs":
            docs = coeus.list_documents()
            if docs:
                for doc in docs:
                    print(f"  - {doc['name']} ({doc['chunks']} chunks)")
            else:
                print("No documents loaded. Put files in ./documents and use /load")
            continue
        if user_input.lower() == "/cleardocs":
            count = coeus.clear_rag_database()
            print(f"Cleared {count} RAG chunks.")
            continue
        if user_input.lower().startswith("/add "):
            path = user_input[5:].strip()
            result = coeus.add_document(path)
            if result.get("success"):
                print(f"Added {result['document']} ({result['chunks_created']} chunks)")
            else:
                print(f"Error: {result.get('error')}")
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