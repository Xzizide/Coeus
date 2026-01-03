import os
import ollama
from dotenv import load_dotenv
from memory import ConversationMemory

load_dotenv()

class Coeus:
    def __init__(self):
        self.model = os.getenv("MODEL_NAME")
        self.base_system_prompt = "You are a helpful assistant named Coeus. You are also short and concise with your answers but you always try to answer truthfully. You are very much into meme culture."
        self.memory = ConversationMemory()

    def _build_system_prompt(self, user_message: str) -> str:
        relevant_memories = self.memory.search_memories(user_message, n_results=5)
        memory_context = self.memory.format_memories_for_prompt(relevant_memories)

        if memory_context:
            return f"{self.base_system_prompt}\n\n{memory_context}"
        return self.base_system_prompt

    def chat(self, message: str):
        system_prompt = self._build_system_prompt(message)

        messages = [
            {
                'role': 'system',
                'content': system_prompt
            }
        ]

        messages.append({'role': 'user', 'content': message})

        stream = ollama.chat(model=self.model, messages=messages, stream=True)

        full_response = ""
        for chunk in stream:
            content = chunk["message"]["content"]
            full_response += content
            yield content

        self.memory.add_memory(message, full_response)

if __name__ == "__main__":
    coeus = Coeus()

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "/clear":
            count = coeus.memory.clear_memories()
            print(f"Cleared {count} memories.")
            continue
        if user_input.lower() == "/count":
            print(f"Memory count: {coeus.memory.get_memory_count()}")
            continue

        print("Coeus: ", end="")
        for chunk in coeus.chat(user_input):
            print(chunk, end="", flush=True)
        print("")