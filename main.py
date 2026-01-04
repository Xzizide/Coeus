from ai_logic.CoeusModel import Coeus


# TTS is loaded lazily to avoid slow startup
tts_engine = None


def get_tts():
    """Lazy load TTS engine on first use."""
    global tts_engine
    if tts_engine is None:
        from ai_logic.tts import VoiceTTS
        tts_engine = VoiceTTS()
    return tts_engine


def main():
    coeus = Coeus()

    print("Coeus initialized with tools:", coeus.tools.list_tools())
    print("Commands: /clear, /reset, /count")
    print("RAG: /load, /docs, /cleardocs, /add <path>")
    print("TTS: /voice, /tts, /notts")

    # Auto-load documents on startup
    result = coeus.load_documents()
    if result.get("loaded"):
        print(f"Loaded {len(result['loaded'])} documents ({result['total_chunks']} chunks)")

    tts_enabled = False

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
        if user_input.lower() == "/voice":
            from ai_logic.tts import setup_voice_interactive
            voice_path = setup_voice_interactive()
            if voice_path:
                tts = get_tts()
                tts.set_speaker_wav(voice_path)
                print(f"Voice configured: {voice_path}")
                tts_enabled = True
                print("TTS enabled.")
            continue

        if user_input.lower() == "/tts":
            tts = get_tts()
            if tts.get_speaker_wav():
                tts_enabled = True
                print("TTS enabled.")
            else:
                print("No voice configured. Run /voice first.")
            continue

        if user_input.lower() == "/notts":
            tts_enabled = False
            print("TTS disabled.")
            continue

        print("Coeus: ", end="")

        # Collect response for TTS
        full_response = ""
        sentence_buffer = ""

        for event in coeus.chat(user_input):
            if event["type"] == "tool_call":
                print(f"\n[Using tool: {event['name']} with {event['args']}]")
            elif event["type"] == "content":
                text = event["text"]
                print(text, end="", flush=True)
                full_response += text

                # Stream TTS on phrases (commas, periods, etc.)
                if tts_enabled:
                    sentence_buffer += text
                    # Break on any natural pause point
                    break_chars = ['. ', '! ', '? ', ', ', ': ', '; ', '.\n', '!\n', '?\n', ',\n']
                    while any(end in sentence_buffer for end in break_chars):
                        # Find the first break point
                        best_idx = len(sentence_buffer)
                        for end in break_chars:
                            idx = sentence_buffer.find(end)
                            if idx != -1 and idx < best_idx:
                                best_idx = idx + len(end)

                        if best_idx < len(sentence_buffer) or best_idx == len(sentence_buffer):
                            phrase = sentence_buffer[:best_idx].strip()
                            sentence_buffer = sentence_buffer[best_idx:]

                            # Only speak if we have enough content (at least 3 words)
                            if phrase and len(phrase.split()) >= 3:
                                try:
                                    tts = get_tts()
                                    tts.speak_and_play(phrase, streaming=False)
                                except Exception as e:
                                    print(f"\n[TTS Error: {e}]")
                            elif phrase:
                                # Too short, put it back
                                sentence_buffer = phrase + " " + sentence_buffer
                                break
                        else:
                            break

        print("")

        # Speak any remaining text in buffer
        if tts_enabled and sentence_buffer.strip():
            try:
                tts = get_tts()
                tts.speak_and_play(sentence_buffer.strip(), streaming=False)
            except Exception as e:
                print(f"[TTS Error: {e}]")


if __name__ == "__main__":
    main()
