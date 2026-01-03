import os
import ollama
from dotenv import load_dotenv

load_dotenv()

class Coeus:
    def __init__(self):
        self.model = os.getenv("MODEL_NAME")
        self.system_prompt = "You are a helpful assistant named Coeus. You have a quirky and a little ironic personality. You are also short and concise with your answers."

    def chat(self, message):
        messages = [
            {
                'role': 'system',
                'content': self.system_prompt
            }
        ]

        messages.append({'role': 'user', 'content': message})

        stream = ollama.chat(model=self.model, messages=messages, stream=True)

        full_response = ""
        for chunk in stream:
            content = chunk["message"]["content"]
            full_response += content
            yield content

if __name__ == "__main__":
    coeus = Coeus()

    while True:
        for chunk in coeus.chat(input("")):
            print(chunk, end="", flush=True)
        print("")