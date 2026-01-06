from RealtimeTTS import TextToAudioStream, KokoroEngine


class VoiceTTS:
    def __init__(self):
        """Initialize RealtimeTTS with system voice."""
        print("Initializing TTS with system voice...")
        self.engine = KokoroEngine(voice="bm_lewis")
        self.stream = TextToAudioStream(self.engine)
        print("TTS ready!")

    def speak(self, text: str):
        """
        Speak text using system voice with real-time streaming.

        Args:
            text: The text to speak
        """
        if not text.strip():
            return

        # Feed text to the stream and play
        self.stream.feed(text)
        self.stream.play()

    def speak_async(self, text: str):
        """
        Speak text asynchronously (non-blocking).

        Args:
            text: The text to speak
        """
        if not text.strip():
            return

        self.stream.feed(text)
        self.stream.play_async()

    def stop(self):
        """Stop current playback."""
        self.stream.stop()

    def is_playing(self) -> bool:
        """Check if TTS is currently playing."""
        return self.stream.is_playing()
