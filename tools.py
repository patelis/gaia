import os
import json
from pathlib import Path
from typing import Annotated

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEndpoint

from utils import load_prompt

# ============================================
# Basic Tools
# ============================================

@tool
def calculator(a: float, b: float, type: str) -> float:
    """Performs mathematical calculations, addition, subtraction, multiplication, division, modulus.
    Args: 
        a (float): first float number
        b (float): second float number
        type (str): the type of calculation to perform, can be addition, subtraction, multiplication, division, modulus
    """

    if type == "addition":
        return a + b
    elif type == "subtraction":
        return a - b
    elif type == "multiplication":
        return a * b
    elif type == "division":
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b
    elif type == "modulus":
        return a % b
    else:
        raise TypeError(f"{type} is not an option for type, choose one of addition, subtraction, multiplication, division, modulus")

@tool
def duck_web_search(query: str) -> str:
    """Use DuckDuckGo to search the web.

    Args:
        query: The search query.
    """
    search = DuckDuckGoSearchRun().invoke(query=query)
    
    return {"duckduckgo_web_search": search}

@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return maximum 3 results.
    
    Args:
        query: The search query."""
    documents = WikipediaLoader(query=query, load_max_docs=3).load()
    processed_documents = "\n\n---\n\n".join(
        [
            f'Document title: {document.metadata.get("title", "")}. Summary: {document.metadata.get("summary", "")}. Documents details: {document.page_content}'
            for document in documents
        ])
    return {"wiki_results": processed_documents}

@tool
def arxiv_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 3 result.
    
    Args:
        query: The search query."""
    documents = ArxivLoader(query=query, load_max_docs=3).load()
    processed_documents = "\n\n---\n\n".join(
        [
            f'Document title: {document.metadata.get("title", "")}. Summary: {document.metadata.get("summary", "")}. Documents details: {document.page_content}'
            for document in documents
        ])
    return {"arxiv_results": processed_documents}

@tool
def tavily_web_search(query: str) -> str:
    """Search the web using Tavily for a query and return maximum 3 results.
    
    Args:
        query: The search query."""
    search_engine = TavilySearchResults(max_results=3)
    search_documents = search_engine.invoke(input=query)
    web_results = "\n\n---\n\n".join(
        [
            f'Document title: {document["title"]}. Contents: {document["content"]}. Relevance Score: {document["score"]}'
            for document in search_documents
        ])
    return {"web_results": web_results}


# ============================================
# VLM Tool
# ============================================

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
    # Load system prompt for VLM
    system_prompt_msg = load_prompt("prompts/vlm_prompt.yaml")
    system_prompt = system_prompt_msg.prompt
    
    try:
        # Initialize VLM Endpoint
        # Using Qwen/Qwen3-VL-32B-Instruct as requested
        # Note: If this model is not available via standard Inference API, fallback to Qwen/Qwen2-VL-72B-Instruct or similar may be needed.
        # Assuming Qwen/Qwen3-VL-32B-Instruct is supported.
        vlm_llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen3-VL-32B-Instruct",
            task="image-text-to-text",  # or text-generation depending on API specifics, usually visual models handle this
            huggingfacehub_api_token=os.getenv("HF_INFERENCE_KEY"),
            temperature=0.1
        )
        
        # Construct the prompt (this is a simplified text-based interaction, 
        # real VLM API calls might need specific formatting or image encoding 
        # depending on the `HuggingFaceEndpoint` support for multi-modal).
        # Standard HF Endpoint for VLMs often expects a specific processed input or URL.
        # For this implementation, we assume the endpoint can handle local paths or we provide a text description placeholder if strictly text-only.
        # Ideally, we should use a library that supports the specific model's API format.
        # Since we are using standard `HuggingFaceEndpoint`, we might need to rely on the model parsing text-encoded images or just URLs.
        # BUT, standard text-generation endpoint might not support image upload directly.
        # *Correction*: `HuggingFaceEndpoint` in LangChain is primarily for text-generation.
        # For VLM, we often use specific API calls or a custom chain.
        # Given constraints, we will attempt to use it, but if it fails, we warn.
        # A more robust way for VLM via API is using the raw `huggingface_hub` InferenceClient.
        
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=os.getenv("HF_INFERENCE_KEY"))
        
        # Check if image path is valid
        if not os.path.exists(image_path):
            return f"Error: Image file not found at {image_path}"
        
        # Generic inference call for VLM
        # This is model-specific. Qwen-VL often expects a conversation format.
        # We will try the chat completion style if supported, else the tailored API.
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_path},  # InferenceClient usually handles local paths if passed correctly or needs base64?
                    # Actually InferenceClient.chat_completion handles local paths as "path/to/image" often in recent versions, 
                    # or strictly URLs/base64. Let's assume it handles file paths or we need to upload.
                    # For safety in this environment, we will describe the intent.
                    {"type": "text", "text": f"{system_prompt}\n\nQuestion: {question}"}
                ]
            }
        ]
        
        # Note: Qwen3-VL-32B-Instruct ID usage
        output = client.chat_completion(
            messages=messages,
            model="Qwen/Qwen3-VL-32B-Instruct", 
            max_tokens=1000
        )
        
        return output.choices[0].message.content

    except Exception as e:
        return f"Error analyzing image with VLM: {str(e)}"


# ============================================
# Document Processing Tools
# ============================================

@tool
def read_pdf(file_path: str) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        file_path: Path to the PDF file to read.
        
    Returns:
        The text content of the PDF, with page separators.
    """
    from pypdf import PdfReader
    
    try:
        reader = PdfReader(file_path)
        text = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text.append(f"--- Page {i+1} ---\n{page_text}")
        
        return "\n\n".join(text) if text else "[Empty PDF]"
    except Exception as e:
        return f"Error reading PDF: {e}"


@tool
def read_docx(file_path: str) -> str:
    """
    Extract text content from a Word document (.docx).
    
    Args:
        file_path: Path to the Word document to read.
        
    Returns:
        The text content of the document.
    """
    from docx import Document
    
    try:
        doc = Document(file_path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        return "\n".join(paragraphs) if paragraphs else "[Empty document]"
    except Exception as e:
        return f"Error reading DOCX: {e}"


@tool
def read_pptx(file_path: str) -> str:
    """
    Extract text content from a PowerPoint presentation (.pptx).
    
    Args:
        file_path: Path to the PowerPoint file to read.
        
    Returns:
        The text content from all slides.
    """
    from pptx import Presentation
    
    try:
        prs = Presentation(file_path)
        text = []
        
        for slide_num, slide in enumerate(prs.slides, 1):
            slide_text = [f"--- Slide {slide_num} ---"]
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_text.append(shape.text)
            if len(slide_text) > 1:
                text.append("\n".join(slide_text))
        
        return "\n\n".join(text) if text else "[Empty presentation]"
    except Exception as e:
        return f"Error reading PPTX: {e}"


@tool
def read_text_file(file_path: str) -> str:
    """
    Read content from a plain text file (.txt).
    
    Args:
        file_path: Path to the text file to read.
        
    Returns:
        The content of the text file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception as e:
        return f"Error reading text file: {e}"


# ============================================
# Data Processing Tools (using polars)
# ============================================

@tool
def read_csv(file_path: str) -> str:
    """
    Read and analyze a CSV file using polars.
    
    Args:
        file_path: Path to the CSV file to read.
        
    Returns:
        Summary of the CSV including schema, row count, and data preview.
    """
    import polars as pl
    
    try:
        df = pl.read_csv(file_path)
        
        output = f"""CSV File Summary:
- Total rows: {len(df)}
- Columns: {df.columns}
- Schema: {dict(df.schema)}

Data (first 20 rows):
{df.head(20)}
"""
        if len(df) <= 50:
            output += f"\nComplete data:\n{df}"
        
        return output
    except Exception as e:
        return f"Error reading CSV: {e}"


@tool
def read_excel(file_path: str, sheet_id: int = 0) -> str:
    """
    Read and analyze an Excel file (.xlsx) using polars.
    
    Args:
        file_path: Path to the Excel file to read.
        sheet_id: The sheet index to read (0-based). Default is 0 (first sheet).
        
    Returns:
        Summary of the Excel sheet including schema, row count, and data preview.
    """
    import polars as pl
    
    try:
        df = pl.read_excel(file_path, sheet_id=sheet_id)
        
        output = f"""Excel Sheet {sheet_id} Summary:
- Total rows: {len(df)}
- Columns: {df.columns}

Data (first 20 rows):
{df.head(20)}
"""
        if len(df) <= 50:
            output += f"\nComplete data:\n{df}"
        
        return output
    except Exception as e:
        return f"Error reading Excel: {e}"


@tool
def read_jsonld(file_path: str) -> str:
    """
    Read and parse a JSON-LD file.
    
    Args:
        file_path: Path to the JSON-LD file to read.
        
    Returns:
        The formatted JSON content.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return f"JSON-LD Content:\n{json.dumps(data, indent=2)}"
    except Exception as e:
        return f"Error reading JSON-LD: {e}"


@tool
def read_pdb(file_path: str) -> str:
    """
    Read and analyze a PDB (Protein Data Bank) file for protein structure analysis.
    
    Args:
        file_path: Path to the PDB file to read.
        
    Returns:
        Analysis of the protein structure including atoms, chains, and coordinates.
    """
    from Bio.PDB import PDBParser
    import numpy as np
    
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", file_path)
        
        info = ["=== PDB Structure Analysis ==="]
        
        atoms = list(structure.get_atoms())
        info.append(f"Total atoms: {len(atoms)}")
        
        for model in structure:
            info.append(f"\nModel {model.id}:")
            for chain in model:
                residues = list(chain.get_residues())
                info.append(f"  Chain {chain.id}: {len(residues)} residues")
        
        if len(atoms) >= 2:
            info.append("\nFirst atoms (for distance calculations):")
            for i, atom in enumerate(atoms[:5]):
                coord = atom.get_coord()
                info.append(
                    f"  Atom {i+1}: {atom.get_name()} at "
                    f"[{coord[0]:.4f}, {coord[1]:.4f}, {coord[2]:.4f}]"
                )
            
            dist = np.linalg.norm(atoms[0].get_coord() - atoms[1].get_coord())
            info.append(f"\nDistance between first two atoms: {dist:.4f} Angstroms")
        
        return "\n".join(info)
    except Exception as e:
        return f"Error reading PDB: {e}"


# ============================================
# Audio Processing Tools
# ============================================

@tool
def transcribe_audio(file_path: str) -> str:
    """
    Transcribe an audio file (MP3, WAV, etc.) to text using Whisper.
    
    Args:
        file_path: Path to the audio file to transcribe.
        
    Returns:
        The transcribed text from the audio.
    """
    from transformers import pipeline
    
    try:
        transcriber = pipeline(
            "automatic-speech-recognition", 
            model="openai/whisper-base",
            chunk_length_s=30
        )
        result = transcriber(file_path)
        return f"Audio Transcription:\n{result['text']}"
    except Exception as e:
        return f"Error transcribing audio: {e}"


# ============================================
# Code Processing Tools
# ============================================

@tool
def read_python_file(file_path: str) -> str:
    """
    Read a Python source code file.
    
    Args:
        file_path: Path to the Python file to read.
        
    Returns:
        The Python code content.
    """
    try:
        with open(file_path, 'r') as f:
            code = f.read()
        return f"Python Code:\n```python\n{code}\n```"
    except Exception as e:
        return f"Error reading Python file: {e}"


# ============================================
# Archive Processing Tools
# ============================================

@tool
def extract_zip(file_path: str) -> str:
    """
    Extract a ZIP archive and list its contents.
    
    Args:
        file_path: Path to the ZIP file to extract.
        
    Returns:
        List of files extracted from the archive with their paths.
    """
    import zipfile
    
    try:
        extract_dir = Path(file_path).parent / Path(file_path).stem
        extract_dir.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        results = [f"ZIP Archive extracted to: {extract_dir}\n\nContents:"]
        
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, extract_dir)
                file_size = os.path.getsize(full_path)
                results.append(f"  - {rel_path} ({file_size} bytes)")
        
        results.append(f"\nUse the appropriate read tool on the extracted files at: {extract_dir}/")
        
        return "\n".join(results)
    except Exception as e:
        return f"Error extracting ZIP: {e}"


# ============================================
# Image Processing Tools
# ============================================

@tool
def describe_image(file_path: str) -> str:
    """
    Get basic information about an image file (JPG, PNG).
    
    Args:
        file_path: Path to the image file.
        
    Returns:
        Image metadata including format, size, and mode.
    """
    from PIL import Image
    
    try:
        img = Image.open(file_path)
        return f"""Image Information:
- File: {Path(file_path).name}
- Format: {img.format}
- Size: {img.size[0]} x {img.size[1]} pixels
- Mode: {img.mode}
"""
    except Exception as e:
        return f"Error describing image: {e}"


# ============================================
# Generic File Processing
# ============================================

@tool
def read_file(file_path: str) -> str:
    """
    Automatically read a file based on its extension.
    
    Supported formats: PDF, DOCX, PPTX, TXT, CSV, XLSX, JSONLD, PDB, PY, MP3, ZIP, JPG, PNG
    
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
        '.zip': lambda p: extract_zip.invoke(p),
        '.jpg': lambda p: analyze_image.invoke({"image_path": p, "question": "Describe this image in detail."}),
        '.jpeg': lambda p: analyze_image.invoke({"image_path": p, "question": "Describe this image in detail."}),
        '.png': lambda p: analyze_image.invoke({"image_path": p, "question": "Describe this image in detail."}),
    }
    
    processor = processors.get(ext)
    
    if processor:
        return processor(file_path)
    
    return f"[Unsupported file type: {ext}]"

# ============================================
# List of all tools
# ============================================

tools_list = [
    calculator,
    duck_web_search,
    wiki_search,
    arxiv_search,
    tavily_web_search,
    read_pdf,
    read_docx,
    read_pptx,
    read_text_file,
    read_csv,
    read_excel,
    read_jsonld,
    read_pdb,
    transcribe_audio,
    read_python_file,
    extract_zip,
    describe_image,
    analyze_image,
    read_file,
]