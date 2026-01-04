import os
import json
import ollama
from dotenv import load_dotenv
from ai_logic.memory import ConversationMemory
from ai_logic.tools import ToolRegistry
from ai_logic.builtin_tools import register_all_builtin_tools
from ai_logic.rag import DocumentRAG

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
        return self.memory.start_new_session()

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

    def chat(self, message: str):
        """
        Chat with streaming response. Yields dictionaries:
        - {"type": "tool_call", "name": str, "args": dict} during tool execution
        - {"type": "content", "text": str} for streamed response chunks
        """
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

                msg = response.get("message", {})
                tool_calls = msg.get("tool_calls")

                if tool_calls:
                    for tc in tool_calls:
                        yield {
                            "type": "tool_call",
                            "name": tc.get("function", {}).get("name"),
                            "args": tc.get("function", {}).get("arguments", {})
                        }

                    messages = self._process_tool_calls(tool_calls, messages)
                    continue

                # No more tool calls - stream the final response
                # Re-query without tools to get streaming
                stream = ollama.chat(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    options={'num_ctx':4096}
                )
                full_response = ""
                for chunk in stream:
                    content = chunk["message"]["content"]
                    full_response += content
                    yield {"type": "content", "text": content}
                self._add_to_history("assistant", full_response)
                self.memory.add_memory(message, full_response)
                return

            else:
                # No tools available - just stream directly
                stream = ollama.chat(model=self.model, messages=messages, stream=True, options={'num_ctx':4096})
                full_response = ""
                for chunk in stream:
                    content = chunk["message"]["content"]
                    full_response += content
                    yield {"type": "content", "text": content}
                self._add_to_history("assistant", full_response)
                self.memory.add_memory(message, full_response)
                return

        yield {"type": "content", "text": "[Max tool iterations reached]"}

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