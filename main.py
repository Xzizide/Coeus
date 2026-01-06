from ai_logic.CoeusModel import Coeus
from ai_logic.tts import VoiceTTS

tts = VoiceTTS()

def main():
    coeus = Coeus()

    print("Coeus initialized with tools:", coeus.tools.list_tools())
    print("Commands: /clear, /reset, /count")
    print("RAG: /load, /docs, /cleardocs, /add <path>")
    print("TTS: /tts, /notts")

    # Auto-load documents on startup
    result = coeus.load_documents()
    if result.get("loaded"):
        print(f"Loaded {len(result['loaded'])} documents ({result['total_chunks']} chunks)")

    tts_enabled = True

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

        # TTS Commands
        if user_input.lower() == "/tts":
            tts_enabled = True
            print("TTS enabled.")
            continue

        if user_input.lower() == "/notts":
            tts_enabled = False
            print("TTS disabled.")
            continue

        print("Coeus: ", end="")

        if tts_enabled:
            try:
                tts.stop()
            except Exception as e:
                print(f"\n[TTS Error: {e}]")
                tts_enabled = False

        full_response = ""
        for event in coeus.chat(user_input):
            if event["type"] == "tool_call":
                print(f"\n[Using tool: {event['name']} with {event['args']}]")
            elif event["type"] == "content":
                text = event["text"]
                print(text, end="", flush=True)
                full_response += text

                # Stream to TTS in real-time
                if tts_enabled:
                    try:
                        tts.stream.feed(text)
                    except Exception as e:
                        print(f"\n[TTS Error: {e}]")
                        tts_enabled = False

        print("")

        # Play the complete response through TTS
        if tts_enabled and full_response.strip():
            try:
                tts.stream.play_async()
            except Exception as e:
                print(f"[TTS Error: {e}]")


if __name__ == "__main__":
    main()
