"""Generic file-reading dispatcher: routes a path to the correct extension-specific tool."""
from pathlib import Path

from langchain_core.tools import tool

from gaia.tools.files import (
    read_pdf, read_docx, read_pptx, read_text_file,
    read_csv, read_excel, read_jsonld, read_pdb,
    read_python_file, extract_zip,
)
from gaia.tools.media import analyze_image, transcribe_audio


@tool
def read_file(file_path: str) -> str:
    """
    Automatically read a file based on its extension.

    Supported formats: PDF, DOCX, PPTX, TXT, CSV, XLSX, JSON-LD, PDB, Python, ZIP, JPG, JPEG, PNG, MP3, WAV, FLAC, OGG, M4A

    Args:
        file_path: Path to the file to read.

    Returns:
        The processed content of the file.
    """
    ext = Path(file_path).suffix.lower()

    processors = {
        '.pdf': lambda p: read_pdf.invoke(p),
        '.docx': lambda p: read_docx.invoke(p),
        '.pptx': lambda p: read_pptx.invoke(p),
        '.txt': lambda p: read_text_file.invoke(p),
        '.csv': lambda p: read_csv.invoke(p),
        '.xlsx': lambda p: read_excel.invoke(p),
        '.jsonld': lambda p: read_jsonld.invoke(p),
        '.pdb': lambda p: read_pdb.invoke(p),
        '.py': lambda p: read_python_file.invoke(p),
        '.mp3': lambda p: transcribe_audio.invoke(p),
        '.wav': lambda p: transcribe_audio.invoke(p),
        '.flac': lambda p: transcribe_audio.invoke(p),
        '.ogg': lambda p: transcribe_audio.invoke(p),
        '.m4a': lambda p: transcribe_audio.invoke(p),
        '.zip': lambda p: extract_zip.invoke(p),
        '.jpg': lambda p: analyze_image.invoke({"image_path": p, "question": "Describe this image in detail."}),
        '.jpeg': lambda p: analyze_image.invoke({"image_path": p, "question": "Describe this image in detail."}),
        '.png': lambda p: analyze_image.invoke({"image_path": p, "question": "Describe this image in detail."}),
    }

    processor = processors.get(ext)

    if processor:
        return processor(file_path)

    return f"[Unsupported file type: {ext}]"
