"""Multi-modal tools: image analysis (VLM) and audio transcription (ASR)."""
import base64
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from huggingface_hub import InferenceClient
from langchain_core.tools import tool

from gaia.utils import load_config, load_prompt


_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_config = load_config()
_vlm_model_name = _config["models"]["vlm"]["model_name"]
_vlm_system_prompt = load_prompt(str(_PROMPTS_DIR / "vlm_prompt.yaml")).content
_asr_model_name = _config["models"]["asr"]["model_name"]
_hf_client = InferenceClient(token=os.getenv("HF_INFERENCE_KEY"))


@tool
def analyze_image(image_path: str, question: str) -> str:
    """
    Analyze an image using a Vision Language Model (VLM) to answer a specific question.

    Args:
        image_path: Path to the image file (JPG, PNG).
        question: The specific question to answer about the image.

    Returns:
        A detailed description or answer based on the visual content.
    """
    try:
        if not os.path.exists(image_path):
            return f"[analyze_image] image file not found at {image_path}"

        with open(image_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode("utf-8")
        ext = Path(image_path).suffix.lower().lstrip(".")
        mime_type = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        image_url = f"data:{mime_type};base64,{image_data}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": f"{_vlm_system_prompt}\n\nQuestion: {question}"}
                ]
            }
        ]

        output = _hf_client.chat_completion(
            messages=messages,
            model=_vlm_model_name,
            max_tokens=1000
        )

        return output.choices[0].message.content

    except Exception as e:
        return f"[analyze_image] VLM call failed: {e}"


@tool
def transcribe_audio(file_path: str) -> str:
    """
    Transcribe an audio file (MP3, WAV, etc.) to text using Whisper.

    Args:
        file_path: Path to the audio file to transcribe.

    Returns:
        The transcribed text from the audio, or a detailed `[transcribe_audio] ...`
        error string identifying file path, size, model, and exception class+message.
    """
    if not os.path.exists(file_path):
        return f"[transcribe_audio] file not found at {file_path}"
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        return f"[transcribe_audio] file is empty at {file_path}"

    try:
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        result = _hf_client.automatic_speech_recognition(audio=audio_bytes, model=_asr_model_name)
        return f"Audio Transcription:\n{result.text}"
    except Exception as e:
        return (
            f"[transcribe_audio] ASR call failed for {file_path} ({file_size} bytes) "
            f"with model '{_asr_model_name}': {type(e).__name__}: {e}"
        )
