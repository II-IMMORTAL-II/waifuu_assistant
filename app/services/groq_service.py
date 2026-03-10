from typing import List, Optional, Iterator
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import logging
import time

from config import (
    GROQ_API_KEYS,
    GROQ_MODEL,
    JARVIS_SYSTEM_PROMPT,
    GENERAL_CHAT_ADDENDUM,
)
from app.services.vector_store import VectorStoreService
from app.utils.time_info import get_time_information
from app.utils.retry import with_retry


logger = logging.getLogger("J.A.R.V.I.S")

GROQ_REQUEST_TIMEOUT = 60

ALL_APIS_FAILED_MESSAGE = (
    "I'm unable to process your request at the moment. "
    "All API services are temporarily unavailable. "
    "Please try again in a few minutes."
)


class AllGroqApisFailedError(Exception):
    pass


# ------------------ HELPERS ------------------ #

def escape_curly_braces(text: str) -> str:
    if not text:
        return text
    return text.replace("{", "{{").replace("}", "}}")


def _is_rate_limit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "429" in str(exc) or "rate limit" in msg or "tokens per day" in msg


def _log_timing(label: str, elapsed: float, extra: str = ""):
    msg = f"[TIMING] {label}: {elapsed:.3f}s"
    if extra:
        msg += f" | {extra}"
    logger.info(msg)


def _mask_api_key(key: str) -> str:
    if not key or len(key) <= 12:
        return "***masked***"
    return f"{key[:8]}...{key[-4:]}"


# ================== GROQ SERVICE ================== #

class GroqService:

    def __init__(self, vector_store_service: VectorStoreService):

        if not GROQ_API_KEYS:
            raise ValueError(
                "No Groq API keys configured. "
                "Set GROQ_API_KEY (and optionally GROQ_API_KEY_2, etc.) in .env"
            )

        self.llms = [
            ChatGroq(
                groq_api_key=key,
                model_name=GROQ_MODEL,
                temperature=0.6,
                request_timeout=GROQ_REQUEST_TIMEOUT,
            )
            for key in GROQ_API_KEYS
        ]

        self.vector_store_service = vector_store_service

        logger.info(
            f"Initialized GroqService with {len(GROQ_API_KEYS)} API key(s)"
        )

    # ---------------- LLM INVOKE ---------------- #

    def _invoke_llm(
        self,
        prompt: ChatPromptTemplate,
        messages: list,
        question: str,
    ) -> str:

        n = len(self.llms)
        last_exc = None
        keys_tried = []

        for i in range(n):
            keys_tried.append(i)
            masked_key = _mask_api_key(GROQ_API_KEYS[i])
            logger.info(f"Trying API key #{i+1}/{n}: {masked_key}")

            def _invoke_with_key():
                chain = prompt | self.llms[i]
                return chain.invoke(
                    {"history": messages, "question": question}
                )

            try:
                response = with_retry(
                    _invoke_with_key,
                    max_retries=2,
                    initial_delay=0.5,
                )

                if i > 0:
                    logger.info(
                        f"Fallback successful: API key #{i+1}/{n}"
                    )

                return response.content

            except Exception as e:
                last_exc = e
                if _is_rate_limit_error(e):
                    logger.warning(
                        f"API key #{i+1}/{n} rate limited: {masked_key}"
                    )
                else:
                    logger.warning(
                        f"API key #{i+1}/{n} failed: {masked_key} - {str(e)[:100]}"
                    )

                if i < n - 1:
                    logger.info("Falling back to next API key...")
                    continue
                break

        masked_all = ", ".join(
            [_mask_api_key(GROQ_API_KEYS[j]) for j in keys_tried]
        )
        logger.error(f"All {n} API key(s) failed. Tried: {masked_all}")

        raise AllGroqApisFailedError(ALL_APIS_FAILED_MESSAGE) from last_exc

    # ---------------- STREAM ---------------- #

    def _stream_llm(
        self,
        prompt: ChatPromptTemplate,
        messages: list,
        question: str,
    ) -> Iterator[str]:

        n = len(self.llms)
        last_exc = None

        for i in range(n):
            masked_key = _mask_api_key(GROQ_API_KEYS[i])
            logger.info(f"Streaming with API key #{i+1}/{n}: {masked_key}")

            try:
                chain = prompt | self.llms[i]
                chunk_count = 0
                stream_start = time.perf_counter()
                first_chunk_time = None

                for chunk in chain.stream(
                    {"history": messages, "question": question}
                ):
                    content = ""

                    if hasattr(chunk, "content"):
                        content = chunk.content or ""
                    elif isinstance(chunk, dict):
                        content = chunk.get("content", "") or ""

                    if isinstance(content, str) and content:
                        if first_chunk_time is None:
                            first_chunk_time = (
                                time.perf_counter() - stream_start
                            )
                            _log_timing("first_chunk", first_chunk_time)

                        chunk_count += 1
                        yield content

                total_stream = time.perf_counter() - stream_start
                _log_timing(
                    "groq_stream_total",
                    total_stream,
                    f"chunks: {chunk_count}",
                )

                if i > 0:
                    logger.info(
                        f"Fallback successful: API key #{i+1}/{n}"
                    )

                return

            except Exception as e:
                last_exc = e
                if _is_rate_limit_error(e):
                    logger.warning(
                        f"API key #{i+1}/{n} rate limited: {masked_key}"
                    )
                else:
                    logger.warning(
                        f"API key #{i+1}/{n} failed: {masked_key} - {str(e)[:100]}"
                    )

                if i < n - 1:
                    logger.info(
                        "Falling back to next API key for stream..."
                    )
                    continue
                break

        logger.error(f"All {n} API key(s) failed during stream.")
        raise AllGroqApisFailedError(ALL_APIS_FAILED_MESSAGE) from last_exc

    # ---------------- PROMPT BUILDER ---------------- #

    def _build_prompt_and_messages(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None,
        extra_system_parts: Optional[List[str]] = None,
        mode_addendum: str = "",
    ):

        context = ""
        context_sources = []
        t0 = time.perf_counter()

        try:
            retriever = self.vector_store_service.get_retriever(k=10)
            context_docs = retriever.invoke(question)

            if context_docs:
                context = "\n".join(
                    [doc.page_content for doc in context_docs]
                )
                context_sources = [
                    doc.metadata.get("source", "unknown")
                    for doc in context_docs
                ]

                logger.info(
                    f"[CONTEXT] Retrieved {len(context_docs)} chunks "
                    f"from sources: {context_sources}"
                )
            else:
                logger.info("[CONTEXT] No relevant chunks found")

        except Exception as retrieval_err:
            logger.warning(
                f"Vector store retrieval failed: {retrieval_err}"
            )

        finally:
            _log_timing("vector_db", time.perf_counter() - t0)

        time_info = get_time_information()

        system_message = JARVIS_SYSTEM_PROMPT
        system_message += f"\n\nCurrent time and date: {time_info}"

        if context:
            system_message += (
                "\n\nRelevant context from your learning data:\n"
                f"{escape_curly_braces(context)}"
            )

        if extra_system_parts:
            system_message += "\n\n" + "\n\n".join(extra_system_parts)

        if mode_addendum:
            system_message += f"\n\n{mode_addendum}"

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        messages = []

        if chat_history:
            for human_msg, ai_msg in chat_history:
                messages.append(HumanMessage(content=human_msg))
                messages.append(AIMessage(content=ai_msg))

        logger.info(
            f"[PROMPT] System length: {len(system_message)} | "
            f"History pairs: {len(chat_history) if chat_history else 0}"
        )

        return prompt, messages

    # ---------------- PUBLIC METHODS ---------------- #

    def get_response(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None,
    ) -> str:

        try:
            prompt, messages = self._build_prompt_and_messages(
                question,
                chat_history,
                mode_addendum=GENERAL_CHAT_ADDENDUM,
            )

            t0 = time.perf_counter()
            result = self._invoke_llm(prompt, messages, question)
            _log_timing("groq_api", time.perf_counter() - t0)

            logger.info(
                f"[RESPONSE] Length: {len(result)} | Preview: {result[:120]}"
            )

            return result

        except AllGroqApisFailedError:
            raise
        except Exception as e:
            raise Exception(
                f"Error getting response from Groq: {str(e)}"
            ) from e

    def stream_response(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None,
    ) -> Iterator[str]:

        try:
            prompt, messages = self._build_prompt_and_messages(
                question,
                chat_history,
                mode_addendum=GENERAL_CHAT_ADDENDUM,
            )

            yield from self._stream_llm(prompt, messages, question)

        except AllGroqApisFailedError:
            raise
        except Exception as e:
            raise Exception(
                f"Error streaming response from Groq: {str(e)}"
            ) from e