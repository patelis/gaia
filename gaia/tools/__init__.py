"""GAIA agent tool roster. Importing this package exposes every tool and the aggregated `tools_list`."""
from gaia.tools.basic import calculator, python_eval
from gaia.tools.web import (
    duck_web_search, tavily_web_search, wiki_search,
    arxiv_search, fetch_webpage, youtube_transcript,
)
from gaia.tools.files import (
    read_pdf, read_docx, read_pptx, read_text_file,
    read_csv, read_excel, read_jsonld, read_pdb,
    read_python_file, extract_zip,
)
from gaia.tools.media import analyze_image, transcribe_audio
from gaia.tools.dispatcher import read_file


tools_list = [
    calculator,
    duck_web_search,
    wiki_search,
    arxiv_search,
    tavily_web_search,
    fetch_webpage,
    youtube_transcript,
    python_eval,
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
    analyze_image,
    read_file,
]
