import os
import re
import json
import wave
import torch
import threading
from pathlib import Path
from typing import Optional, Generator
import torch

# 1. Import all nested config classes used by XTTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

# 2. Register them all at once
torch.serialization.add_safe_globals([
    XttsConfig, 
    XttsAudioConfig, 
    XttsArgs, 
    BaseDatasetConfig
])

ffmpeg_bin_path = r"C:\FFMPEG\bin" 

if os.path.exists(ffmpeg_bin_path):
    os.add_dll_directory(ffmpeg_bin_path)

from TTS.api import TTS


CONFIG_FILE = "./tts_config.json"
VOICE_SAMPLES_DIR = "./voice_samples"


class VoiceTTS:
    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading TTS model on {self.device}...")
        self.tts = TTS(model_name).to(self.device)
        self.config = self._load_config()
        self._playback_lock = threading.Lock()

        # Ensure voice samples directory exists
        Path(VOICE_SAMPLES_DIR).mkdir(exist_ok=True)

    def _load_config(self) -> dict:
        if Path(CONFIG_FILE).exists():
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        return {}

    def _save_config(self):
        with open(CONFIG_FILE, "w") as f:
            json.dump(self.config, f, indent=2)

    def get_speaker_wav(self) -> Optional[str]:
        """Get the configured speaker wav file path."""
        path = self.config.get("speaker_wav")
        if path and Path(path).exists():
            return path
        return None

    def set_speaker_wav(self, path: str):
        """Set the speaker wav file for voice cloning."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Speaker wav not found: {path}")
        self.config["speaker_wav"] = str(Path(path).resolve())
        self._save_config()

    def speak(self, text: str, output_file: str = "output.wav", speaker_wav: Optional[str] = None) -> str:
        """
        Generate speech from text using voice cloning.

        Args:
            text: The text to convert to speech
            output_file: Path to save the generated audio
            speaker_wav: Path to speaker voice sample (uses config default if not provided)

        Returns:
            Path to the generated audio file
        """
        speaker = speaker_wav or self.get_speaker_wav()

        if not speaker:
            raise ValueError("No speaker wav configured. Use set_speaker_wav() or record_voice_sample() first.")

        # Clean text for TTS
        text = self._clean_text(text)

        if not text.strip():
            return output_file

        self.tts.tts_to_file(
            text=text,
            file_path=output_file,
            speaker_wav=speaker,
            language="en"
        )

        return output_file

    def _clean_text(self, text: str) -> str:
        """Clean text for TTS processing."""
        # Remove markdown formatting
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.+?)\*', r'\1', text)      # Italic
        text = re.sub(r'`(.+?)`', r'\1', text)        # Code
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)  # Links

        # Remove emojis and special characters that TTS can't handle
        text = re.sub(r'[^\w\s.,!?;:\'\"-]', '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def split_sentences(self, text: str) -> list:
        """Split text into sentences for streaming TTS."""
        # Split on sentence endings, keeping the punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Filter out empty strings and very short fragments
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]

    def speak_streaming(self, text: str, speaker_wav: Optional[str] = None) -> Generator[str, None, None]:
        """
        Generate and play speech sentence by sentence for lower perceived latency.

        Args:
            text: The full text to convert to speech
            speaker_wav: Path to speaker voice sample

        Yields:
            Path to each generated audio file
        """
        speaker = speaker_wav or self.get_speaker_wav()

        if not speaker:
            raise ValueError("No speaker wav configured.")

        sentences = self.split_sentences(text)

        for i, sentence in enumerate(sentences):
            sentence = self._clean_text(sentence)
            if not sentence:
                continue

            output_file = f"./temp_tts_{i}.wav"

            self.tts.tts_to_file(
                text=sentence,
                file_path=output_file,
                speaker_wav=speaker,
                language="en"
            )

            yield output_file

    def speak_and_play(self, text: str, speaker_wav: Optional[str] = None, streaming: bool = True):
        """
        Generate speech and play it immediately.

        Args:
            text: Text to speak
            speaker_wav: Path to speaker voice sample
            streaming: If True, play sentence by sentence for lower latency
        """
        if streaming:
            for audio_file in self.speak_streaming(text, speaker_wav):
                self.play_audio(audio_file)
                # Clean up temp file after playing
                try:
                    os.remove(audio_file)
                except:
                    pass
        else:
            output_file = "./temp_tts_full.wav"
            self.speak(text, output_file, speaker_wav)
            self.play_audio(output_file)
            try:
                os.remove(output_file)
            except:
                pass

    def play_audio(self, file_path: str):
        """Play a WAV audio file through the speakers."""
        with self._playback_lock:
            try:
                # Try pygame first (most reliable cross-platform)
                try:
                    import pygame
                    if not pygame.mixer.get_init():
                        pygame.mixer.init()
                    pygame.mixer.music.load(file_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                    return
                except ImportError:
                    pass

                # Try sounddevice + soundfile
                try:
                    import sounddevice as sd
                    import soundfile as sf
                    data, samplerate = sf.read(file_path)
                    sd.play(data, samplerate)
                    sd.wait()
                    return
                except ImportError:
                    pass

                # Try playsound
                try:
                    from playsound import playsound
                    playsound(file_path)
                    return
                except ImportError:
                    pass

                # Fallback to system command (Windows)
                import platform
                if platform.system() == "Windows":
                    import winsound
                    winsound.PlaySound(file_path, winsound.SND_FILENAME)
                    return

                # Linux fallback
                os.system(f'aplay "{file_path}" 2>/dev/null || paplay "{file_path}" 2>/dev/null')

            except Exception as e:
                print(f"[TTS] Playback error: {e}")


def record_voice_sample(duration: int = 8, sample_rate: int = 22050) -> str:
    """
    Guide user to record a voice sample for cloning.

    Args:
        duration: Recording duration in seconds (6-10 recommended)
        sample_rate: Audio sample rate

    Returns:
        Path to the saved voice sample
    """
    try:
        import sounddevice as sd
        from scipy.io import wavfile
    except ImportError:
        print("Required packages not installed. Run: pip install sounddevice scipy")
        return ""

    output_path = Path(VOICE_SAMPLES_DIR) / "my_voice.wav"

    print("\n" + "=" * 50)
    print("VOICE SAMPLE RECORDING")
    print("=" * 50)
    print(f"\nYou'll record {duration} seconds of your voice.")
    print("\nTips for best results:")
    print("  - Speak naturally and clearly")
    print("  - Read a paragraph of text or describe something")
    print("  - Avoid background noise")
    print("  - Keep consistent volume")
    print("\nPress ENTER when ready to start recording...")
    input()

    print(f"\nRecording for {duration} seconds... SPEAK NOW!")

    try:
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()

        print("Recording complete!")

        # Save the recording
        wavfile.write(str(output_path), sample_rate, audio)
        print(f"Saved to: {output_path}")

        # Optionally play back
        print("\nPlay back recording? (y/n): ", end="")
        if input().lower() == 'y':
            print("Playing...")
            sd.play(audio, sample_rate)
            sd.wait()

        print("\nKeep this recording? (y/n): ", end="")
        if input().lower() != 'y':
            os.remove(output_path)
            print("Recording discarded. Run again to re-record.")
            return ""

        return str(output_path)

    except Exception as e:
        print(f"Recording error: {e}")
        return ""


def setup_voice_interactive() -> Optional[str]:
    """Interactive setup for voice cloning."""
    print("\n" + "=" * 50)
    print("VOICE CLONING SETUP")
    print("=" * 50)
    print("\nOptions:")
    print("  1. Record new voice sample")
    print("  2. Use existing WAV file")
    print("  3. Skip (no TTS)")

    choice = input("\nChoice (1/2/3): ").strip()

    if choice == "1":
        return record_voice_sample()
    elif choice == "2":
        path = input("Enter path to WAV file: ").strip()
        if Path(path).exists():
            return path
        print("File not found.")
        return None
    else:
        return None
