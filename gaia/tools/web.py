"""Web search and fetching tools: DuckDuckGo, Tavily, Wikipedia, Arxiv, webpage fetch, YouTube transcripts."""
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
from langchain_core.tools import tool

from gaia.utils import extract_youtube_id, load_config, download_task_file


_ddg_search = None
_tavily_search = None


def _get_ddg():
    global _ddg_search
    if _ddg_search is None:
        _ddg_search = DuckDuckGoSearchRun()
    return _ddg_search


def _get_tavily():
    global _tavily_search
    if _tavily_search is None:
        _tavily_search = TavilySearchResults(max_results=3)
    return _tavily_search


@tool
def duck_web_search(query: str) -> str:
    """Use DuckDuckGo to search the web.

    Args:
        query: The search query.
    """
    search = _get_ddg().invoke(input=query)
    return {"duckduckgo_web_search": search}


@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return maximum 3 results.

    Args:
        query: The search query."""
    documents = WikipediaLoader(query=query, load_max_docs=3, doc_content_chars_max=20000).load()
    processed_documents = "\n\n---\n\n".join(
        [
            f'Document title: {document.metadata.get("title", "")}. Summary: {document.metadata.get("summary", "")}. Documents details: {document.page_content}'
            for document in documents
        ])
    return {"wiki_results": processed_documents}


@tool
def wikipedia_page_fetch(title: str) -> str:
    """Fetch the full text of a Wikipedia page by its exact title.

    Use this when you can guess the page title from the question (e.g., "1928 Summer
    Olympics", "List of Featured Articles"). Faster than search + fetch_webpage when
    the title is obvious. Returns full page content, not a search summary.

    Args:
        title: The exact Wikipedia page title (including namespace prefix if applicable,
               e.g., "Wikipedia:Featured_article_candidates/Featured_log/November_2016").

    Returns:
        The page content prefixed with title and URL, or a `[wikipedia_page_fetch] ...`
        error string.
    """
    import wikipedia
    try:
        page = wikipedia.page(title, auto_suggest=False)
        return f"Wikipedia: {page.title}\nURL: {page.url}\n\n{page.content}"
    except wikipedia.exceptions.DisambiguationError as e:
        return f"[wikipedia_page_fetch] '{title}' is a disambiguation page. Options: {e.options[:10]}"
    except wikipedia.exceptions.PageError:
        return f"[wikipedia_page_fetch] page not found: '{title}'. Try wiki_search to find the correct title."
    except Exception as e:
        return f"[wikipedia_page_fetch] failed: {e}"


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
    search_documents = _get_tavily().invoke(input=query)
    web_results = "\n\n---\n\n".join(
        [
            f'Document title: {document["title"]}. Contents: {document["content"]}. Relevance Score: {document["score"]}'
            for document in search_documents
        ])
    return {"web_results": web_results}


@tool
def fetch_webpage(url: str) -> str:
    """
    Fetch and extract the main text content from a webpage.
    Use this when a search result points to a specific URL you need to read in full.

    Args:
        url: The full URL of the page to fetch.

    Returns:
        The extracted text content of the page.
    """
    import trafilatura
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return f"[fetch_webpage] could not fetch {url}"
        text = trafilatura.extract(downloaded, include_tables=True, include_links=False)
        if text is None:
            return f"[fetch_webpage] could not extract content from {url}"
        return f"Page content from {url}:\n\n{text}"
    except Exception as e:
        return f"[fetch_webpage] failed: {e}"


@tool
def retry_file_download(task_id: str, file_name: str) -> str:
    """Retry downloading the task file from the GAIA scoring API.

    Use this when the initial automatic download failed (you will see a message like
    "the automatic download failed" in the question context). Returns the local path
    on success, or an error string starting with `[retry_file_download]`.

    Args:
        task_id: The task ID for the current question.
        file_name: The original file name from the question metadata.

    Returns:
        Local filesystem path to the downloaded file, or an error description.
    """
    cfg = load_config()
    local_path, err = download_task_file(
        task_id=task_id,
        file_name=file_name,
        base_url=cfg["api"]["base_url"],
        files_dir=cfg["api"]["files_dir"],
    )
    if local_path:
        return local_path
    return f"[retry_file_download] {err}"


@tool
def youtube_transcript(url: str) -> str:
    """Fetch the transcript (captions) of a YouTube video as plain text.

    Use this whenever a question references a YouTube URL — the spoken content of
    the video is available via captions. Note: this returns text only; questions
    that require visual analysis of the frames cannot be answered from the
    transcript alone.

    Prefers manually-written English captions; falls back to auto-generated English,
    and finally to any available language.

    Args:
        url: The full YouTube URL (watch, youtu.be, embed, shorts) or a bare 11-char video ID.

    Returns:
        The concatenated transcript text, or an error string starting with `[youtube_transcript]`.
    """
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (
        TranscriptsDisabled, NoTranscriptFound, VideoUnavailable,
    )

    video_id = extract_youtube_id(url)
    if not video_id:
        return f"[youtube_transcript] could not parse video ID from: {url}"

    try:
        ytt_api = YouTubeTranscriptApi()
        try:
            fetched = ytt_api.fetch(video_id, languages=['en'])
        except NoTranscriptFound:
            transcript_list = ytt_api.list(video_id)
            transcript = next(iter(transcript_list))
            fetched = transcript.fetch()

        text = " ".join(snippet.text for snippet in fetched)
        return f"YouTube transcript for {url}:\n\n{text}"
    except TranscriptsDisabled:
        return f"[youtube_transcript] transcripts are disabled for {url}"
    except VideoUnavailable:
        return f"[youtube_transcript] video unavailable: {url}"
    except NoTranscriptFound:
        return f"[youtube_transcript] no transcript found for {url}"
    except Exception as e:
        return f"[youtube_transcript] failed: {e}"
