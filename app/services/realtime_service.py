from typing import List, Optional, Iterator, Tuple, Any
from tavily import TavilyClient
import logging
import os
import time

from app.services.groq_service import (
    GroqService,
    escape_curly_braces,
    AllGroqApisFailedError,
)
from app.services.vector_store import VectorStoreService
from app.utils.retry import with_retry
from config import REALTIME_CHAT_ADDENDUM, GROQ_API_KEYS, GROQ_MODEL


logger = logging.getLogger("J.A.R.V.I.S")

GROQ_REQUEST_TIMEOUT_FAST = 15

_QUERY_EXTRACTION_PROMPT = (
    "You are a search query optimizer. Given the user's message and recent "
    "conversation, produce a single short, focused web search query "
    "(max 12 words) that will find the information the user needs. "
    "Resolve references like 'that website', 'him', 'it' using conversation. "
    "Output ONLY the search query."
)


class RealtimeGroqService(GroqService):

    def __init__(self, vector_store_service: VectorStoreService):
        super().__init__(vector_store_service)

        tavily_api_key = os.getenv("TAVILY_API_KEY", "")

        if tavily_api_key:
            self.tavily_client = TavilyClient(api_key=tavily_api_key)
            logger.info("Tavily search client initialized successfully")
        else:
            self.tavily_client = None
            logger.warning("TAVILY_API_KEY not set. Realtime search unavailable.")

        if GROQ_API_KEYS:
            from langchain_groq import ChatGroq

            self._fast_llm = ChatGroq(
                groq_api_key=GROQ_API_KEYS[0],
                model_name=GROQ_MODEL,
                temperature=0.0,
                request_timeout=GROQ_REQUEST_TIMEOUT_FAST,
                max_tokens=50,
            )
        else:
            self._fast_llm = None

    # ---------------- QUERY EXTRACTION ---------------- #

    def _extract_search_query(
        self,
        question: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
    ) -> str:

        if not self._fast_llm:
            return question

        try:
            t0 = time.perf_counter()

            history_context = ""
            if chat_history:
                recent = chat_history[-3:]
                parts = []
                for human, ai in recent:
                    parts.append(f"User: {human[:200]}")
                    parts.append(f"Assistant: {ai[:200]}")
                history_context = "\n".join(parts)

            if history_context:
                full_prompt = (
                    f"{_QUERY_EXTRACTION_PROMPT}\n\n"
                    f"Recent conversation:\n{history_context}\n\n"
                    f"User's latest message: {question}\n\n"
                    f"Search query:"
                )
            else:
                full_prompt = (
                    f"{_QUERY_EXTRACTION_PROMPT}\n\n"
                    f"User's message: {question}\n\n"
                    f"Search query:"
                )

            response = self._fast_llm.invoke(full_prompt)
            extracted = response.content.strip().strip('"').strip("'")

            if 3 <= len(extracted) <= 200:
                logger.info(
                    "[REALTIME] Query extraction: '%s' -> '%s' (%.3fs)",
                    question[:80],
                    extracted[:80],
                    time.perf_counter() - t0,
                )
                return extracted

            logger.warning("Query extraction unusable, using raw question")
            return question

        except Exception as e:
            logger.warning("Query extraction failed (%s)", e)
            return question

    # ---------------- TAVILY SEARCH ---------------- #

    def search_tavily(
        self,
        query: str,
        num_results: int = 7,
    ) -> Tuple[str, Optional[dict]]:

        if not self.tavily_client:
            logger.warning("Tavily client not initialized.")
            return "", None

        try:
            t0 = time.perf_counter()

            response = with_retry(
                lambda: self.tavily_client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=num_results,
                    include_answer=True,
                    include_raw_content=False,
                ),
                max_retries=3,
                initial_delay=1.0,
            )

            results = response.get("results", [])
            ai_answer = response.get("answer", "")

            if not results and not ai_answer:
                return "", None

            payload = {
                "query": query,
                "answer": ai_answer,
                "results": [
                    {
                        "title": r.get("title", "No title"),
                        "content": (r.get("content") or "")[:500],
                        "url": r.get("url", ""),
                        "score": round(float(r.get("score", 0)), 2),
                    }
                    for r in results[:num_results]
                ],
            }

            parts = [f"=== WEB SEARCH RESULTS FOR: {query} ===\n"]

            if ai_answer:
                parts.append(
                    f"AI-SYNTHESIZED ANSWER:\n{ai_answer}\n"
                )

            if results:
                parts.append("INDIVIDUAL SOURCES:")
                for i, r in enumerate(results[:num_results], 1):
                    parts.append(
                        f"\n[Source {i}] (score: {r.get('score', 0):.2f})"
                    )
                    parts.append(f"Title: {r.get('title')}")
                    if r.get("content"):
                        parts.append(f"Content: {r.get('content')}")
                    if r.get("url"):
                        parts.append(f"URL: {r.get('url')}")

            parts.append("\n=== END SEARCH RESULTS ===")

            formatted = "\n".join(parts)

            logger.info(
                "[TAVILY] %d results | formatted %d chars (%.3fs)",
                len(results),
                len(formatted),
                time.perf_counter() - t0,
            )

            return formatted, payload

        except Exception as e:
            logger.error("Error performing Tavily search: %s", e)
            return "", None

    # ---------------- REALTIME RESPONSE ---------------- #

    def get_response(
        self,
        question: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
    ) -> str:

        try:
            search_query = self._extract_search_query(question, chat_history)

            formatted_results, _ = self.search_tavily(search_query)

            extra_parts = (
                [escape_curly_braces(formatted_results)]
                if formatted_results
                else None
            )

            prompt, messages = self._build_prompt_and_messages(
                question,
                chat_history,
                extra_system_parts=extra_parts,
                mode_addendum=REALTIME_CHAT_ADDENDUM,
            )

            t0 = time.perf_counter()
            response = self._invoke_llm(prompt, messages, question)

            logger.info(
                "[REALTIME RESPONSE] %d chars (%.3fs)",
                len(response),
                time.perf_counter() - t0,
            )

            return response

        except AllGroqApisFailedError:
            raise
        except Exception as e:
            logger.error("Error in realtime get_response: %s", e, exc_info=True)
            raise

    # ---------------- REALTIME STREAM ---------------- #

    def stream_response(
        self,
        question: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
    ) -> Iterator[Any]:

        try:
            search_query = self._extract_search_query(question, chat_history)

            formatted_results, payload = self.search_tavily(search_query)

            if payload:
                yield {"_search_results": payload}

            extra_parts = (
                [escape_curly_braces(formatted_results)]
                if formatted_results
                else None
            )

            prompt, messages = self._build_prompt_and_messages(
                question,
                chat_history,
                extra_system_parts=extra_parts,
                mode_addendum=REALTIME_CHAT_ADDENDUM,
            )

            yield from self._stream_llm(prompt, messages, question)

            logger.info("Realtime stream completed for: %s", search_query)

        except AllGroqApisFailedError:
            raise
        except Exception as e:
            logger.error("Error in realtime stream_response: %s", e, exc_info=True)
            raise