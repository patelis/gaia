"""Web search and fetching tools: DuckDuckGo, Tavily, Wikipedia, Arxiv, webpage fetch, YouTube transcripts."""
import re
from datetime import datetime

import requests
import trafilatura
import wikipedia
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
from langchain_core.tools import tool
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

from gaia.utils import extract_youtube_id, load_config, download_task_file

# Wikipedia blocks/throttles requests with the default `wikipedia` package UA, which
# causes the API to return a non-JSON body and `requests.json()` to raise a
# `JSONDecodeError: Expecting value: line 1 column 1 (char 0)`. Setting an identifying
# UA per Wikipedia's policy fixes this for both `wiki_search` and `wikipedia_page_fetch`.
_USER_AGENT = "gaia-agent/0.1 (https://huggingface.co/spaces/KPatelis/Agents_Course_Assignment)"
wikipedia.set_user_agent(_USER_AGENT)


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
    try:
        search = _get_ddg().invoke(input=query)
        return {"duckduckgo_web_search": search}
    except Exception as e:
        return f"[duck_web_search] failed: {type(e).__name__}: {e}"


@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return up to 3 distinct articles.

    Args:
        query: The search query."""
    try:
        documents = WikipediaLoader(query=query, load_max_docs=3, doc_content_chars_max=20000).load()
        # Deduplicate by article title
        seen_titles = set()
        unique_documents = []
        for d in documents:
            title = d.metadata.get("title", "")
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_documents.append(d)
        processed_documents = "\n\n---\n\n".join(
            [
                f'Document title: {document.metadata.get("title", "")}. Summary: {document.metadata.get("summary", "")}. Documents details: {document.page_content}'
                for document in unique_documents
            ])
        return {"wiki_results": processed_documents}
    except Exception as e:
        return f"[wiki_search] failed: {type(e).__name__}: {e}"


_NAVBOX_MIN_CHARS = 200    # ignore navboxes with less than this many chars of text
_NAVBOX_MAX_CHARS = 15000  # cap navbox text to avoid blowing up context on huge pages


def _extract_navbox_text(html: str) -> str:
    """Pull a flat-text dump of every ``.navbox`` div on a Wikipedia page.

    Navboxes are the cross-link tables Wikipedia puts at the bottom of articles.
    We collect every navbox on the page, flatten whitespace, and join with blank lines. 
    Returns ``""`` if no meaningful navbox content is present.
    """
    soup = BeautifulSoup(html, "html.parser")
    parts = []
    for nb in soup.find_all("div", class_="navbox"):
        text = re.sub(r"\s+", " ", nb.get_text(" ", strip=True))
        if text:
            parts.append(text)
    joined = "\n\n".join(parts).strip()
    if len(joined) < _NAVBOX_MIN_CHARS:
        return ""
    return joined[:_NAVBOX_MAX_CHARS]


@tool
def wikipedia_page_fetch(title: str) -> str:
    """Fetch a Wikipedia page by title and return its body + navbox text.
    Args:
        title: The exact Wikipedia page title, optionally with a namespace prefix
            (e.g. ``"Wikipedia:Featured article candidates/Featured log/November 2016"``).

    Returns:
        On success: a multi-line string starting with ``"Wikipedia: <resolved title>"``,
        a ``URL:`` line, a blank line, the extracted body, and (if present) a
        ``--- Related (navbox) ---`` block.
        On failure: a string starting with ``[wikipedia_page_fetch] …`` describing
        the failure (page not found, disambiguation page, search fallback exhausted).
    """

    def _render(page, resolved_from=None):
        suffix = f" (resolved from '{resolved_from}')" if resolved_from else ""
        header = f"Wikipedia: {page.title}{suffix}\nURL: {page.url}"

        # Body: prefer trafilatura (preserves lists and tables — critical for
        # counting-style questions). Fall back to page.content on failure.
        body = None
        downloaded = trafilatura.fetch_url(page.url)
        if downloaded is not None:
            body = trafilatura.extract(downloaded, include_tables=True, include_links=False)
        if not body:
            body = page.content

        # Navbox: append the cross-link tables that body extractors strip.
        navbox_section = ""
        try:
            navbox_text = _extract_navbox_text(page.html())
            if navbox_text:
                navbox_section = f"\n\n--- Related (navbox) ---\n{navbox_text}"
        except Exception:
            pass

        return f"{header}\n\n{body}{navbox_section}"

    try:
        page = wikipedia.page(title, auto_suggest=False)
        return _render(page)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"[wikipedia_page_fetch] '{title}' is a disambiguation page. Options: {e.options[:10]}"
    except wikipedia.exceptions.PageError:
        # Recover from case-sensitivity / slight title mismatches by searching once and
        # fetching the top hit.
        try:
            hits = wikipedia.search(title, results=1)
        except Exception as e:
            return f"[wikipedia_page_fetch] page not found: '{title}'; search fallback failed: {e}"
        if not hits:
            return f"[wikipedia_page_fetch] page not found: '{title}' and no search hits."
        resolved = hits[0]
        if resolved == title:
            return f"[wikipedia_page_fetch] page not found: '{title}'. Try wiki_search to find the correct title."
        try:
            page = wikipedia.page(resolved, auto_suggest=False)
        except Exception as e:
            return f"[wikipedia_page_fetch] resolved title '{resolved}' but fetch failed: {e}"
        return _render(page, resolved_from=title)
    except Exception as e:
        return f"[wikipedia_page_fetch] failed: {e}"


_WIKI_API_ENDPOINT = "https://en.wikipedia.org/w/api.php"


def _resolve_revision_at(title: str, iso_timestamp: str) -> tuple[int | None, str | None, str | None]:
    """Look up the Wikipedia revision id active for ``title`` at ``iso_timestamp``.
    """
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "titles": title,
        "rvprop": "ids|timestamp",
        "rvlimit": 1,
        "rvdir": "older",
        "rvstart": iso_timestamp,
    }
    try:
        r = requests.get(
            _WIKI_API_ENDPOINT,
            params=params,
            headers={"User-Agent": _USER_AGENT},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return None, None, f"API request failed: {type(e).__name__}: {e}"

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None, None, "API returned no pages"
    page = next(iter(pages.values()))
    if "missing" in page:
        return None, None, f"page not found: '{title}'"
    revisions = page.get("revisions") or []
    if not revisions:
        return None, None, f"no revisions for '{title}' on or before {iso_timestamp}"
    return revisions[0]["revid"], page.get("title", title), None


@tool
def wikipedia_page_as_of(title: str, date: str) -> str:
    """Fetch a Wikipedia page as it existed at end of day UTC on a specific date.
    Args:
        title: Wikipedia page title (e.g. ``"Taishō Tamai"``,
            ``"Hokkaido Nippon-Ham Fighters"``, ``"1928 Summer Olympics"``).
        date: Target date in ISO ``"YYYY-MM-DD"`` format (e.g. ``"2023-07-31"``).
            The page is fetched as it appeared at 23:59:59 UTC on that day.

    Returns:
        On success: a multi-line string ``"Wikipedia: <title> (as of <date>, revid <id>) / URL: <oldid URL> / <body> / --- Related (navbox) ---"``.
        On failure: a string starting with ``[wikipedia_page_as_of] …`` describing
        the failure (invalid date, page not found, revision lookup failure,
        rendered-HTML fetch failure).
    """
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return f"[wikipedia_page_as_of] invalid date '{date}'; expected YYYY-MM-DD."
    iso_ts = dt.strftime("%Y-%m-%dT23:59:59Z")

    revid, resolved_title, err = _resolve_revision_at(title, iso_ts)
    if err and err.startswith("page not found"):
        # Case-/spelling-tolerant fallback: search and retry the top hit.
        try:
            hits = wikipedia.search(title, results=1)
        except Exception as e:
            return f"[wikipedia_page_as_of] page not found and search failed: {e}"
        if not hits or hits[0] == title:
            return f"[wikipedia_page_as_of] page not found: '{title}'"
        revid, resolved_title, err = _resolve_revision_at(hits[0], iso_ts)
    if err:
        return f"[wikipedia_page_as_of] {err}"

    url = f"https://en.wikipedia.org/w/index.php?oldid={revid}"
    try:
        resp = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=30)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        return f"[wikipedia_page_as_of] could not fetch revision URL {url}: {type(e).__name__}: {e}"

    body = trafilatura.extract(html, include_tables=True, include_links=False)
    if not body:
        return f"[wikipedia_page_as_of] no body extracted from {url}"

    navbox_section = ""
    try:
        navbox_text = _extract_navbox_text(html)
        if navbox_text:
            navbox_section = f"\n\n--- Related (navbox) ---\n{navbox_text}"
    except Exception:
        pass

    header = f"Wikipedia: {resolved_title} (as of {date}, revid {revid})\nURL: {url}"
    return f"{header}\n\n{body}{navbox_section}"


@tool
def arxiv_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 3 result.

    Args:
        query: The search query."""
    try:
        documents = ArxivLoader(query=query, load_max_docs=3).load()
        processed_documents = "\n\n---\n\n".join(
            [
                f'Document title: {document.metadata.get("title", "")}. Summary: {document.metadata.get("summary", "")}. Documents details: {document.page_content}'
                for document in documents
            ])
        return {"arxiv_results": processed_documents}
    except Exception as e:
        return f"[arxiv_search] failed: {type(e).__name__}: {e}"


@tool
def tavily_web_search(query: str) -> str:
    """Search the web using Tavily for a query and return maximum 3 results.

    Args:
        query: The search query."""
    try:
        search_documents = _get_tavily().invoke(input=query)
        web_results = "\n\n---\n\n".join(
            [
                f'Document title: {document["title"]}. Contents: {document["content"]}. Relevance Score: {document["score"]}'
                for document in search_documents
            ])
        return {"web_results": web_results}
    except Exception as e:
        return f"[tavily_web_search] failed: {type(e).__name__}: {e}"


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
    Args:
        url: The full YouTube URL (watch, youtu.be, embed, shorts) or a bare 11-char video ID.

    Returns:
        The concatenated transcript text, or an error string starting with `[youtube_transcript]`.
    """

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
