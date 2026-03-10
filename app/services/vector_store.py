import json
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("J.A.R.V.I.S")

# Heavy ML imports — optional so the app starts even if torch/faiss crash
_VECTOR_AVAILABLE = False
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    _VECTOR_AVAILABLE = True
except Exception as e:
    logger.warning("[VECTOR] ML imports failed (%s). Vector store disabled.", e)

from config import (
    LEARNING_DATA_DIR,
    CHATS_DATA_DIR,
    VECTOR_STORE_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

logger = logging.getLogger("J.A.R.V.I.S")


class VectorStoreService:

    def __init__(self):
        if not _VECTOR_AVAILABLE:
            logger.warning("[VECTOR] Running without vector store — ML libs unavailable.")
            self.embeddings = None
            self.text_splitter = None
            self.vector_store = None
            self._retriever_cache = {}
            return

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        self.vector_store: Optional[FAISS] = None
        self._retriever_cache: dict = {}

    # ---------------- LOAD LEARNING DATA ---------------- #

    def load_learning_data(self) -> List[Document]:
        documents = []

        for file_path in sorted(LEARNING_DATA_DIR.glob("*.txt")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                if content:
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={"source": str(file_path.name)},
                        )
                    )
                    logger.info(
                        "[VECTOR] Loaded learning data: %s (%d chars)",
                        file_path.name,
                        len(content),
                    )

            except Exception as e:
                logger.warning(
                    "Could not load learning data file %s: %s",
                    file_path,
                    e,
                )

        logger.info(
            "[VECTOR] Total learning data files loaded: %d",
            len(documents),
        )
        return documents

    # ---------------- LOAD CHAT HISTORY ---------------- #

    def load_chat_history(self) -> List[Document]:
        documents = []

        for file_path in sorted(CHATS_DATA_DIR.glob("*.json")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    chat_data = json.load(f)

                messages = chat_data.get("messages", [])

                chat_content = "\n".join(
                    f"User: {msg.get('content', '')}"
                    if msg.get("role") == "user"
                    else f"Assistant: {msg.get('content', '')}"
                    for msg in messages
                )

                if chat_content.strip():
                    documents.append(
                        Document(
                            page_content=chat_content,
                            metadata={"source": f"chat_{file_path.stem}"},
                        )
                    )

                    logger.info(
                        "[VECTOR] Loaded chat history: %s (%d messages)",
                        file_path.name,
                        len(messages),
                    )

            except Exception as e:
                logger.warning(
                    "Could not load chat history file %s: %s",
                    file_path,
                    e,
                )

        logger.info(
            "[VECTOR] Total chat history files loaded: %d",
            len(documents),
        )
        return documents

    # ---------------- CREATE VECTOR STORE ---------------- #

    def create_vector_store(self):
        if not _VECTOR_AVAILABLE:
            logger.warning("[VECTOR] Skipping vector store creation — ML libs unavailable.")
            return None
        learning_docs = self.load_learning_data()
        chat_docs = self.load_chat_history()

        all_documents = learning_docs + chat_docs

        logger.info(
            "[VECTOR] Total documents to index: %d (learning: %d, chat: %d)",
            len(all_documents),
            len(learning_docs),
            len(chat_docs),
        )

        if not all_documents:
            self.vector_store = FAISS.from_texts(
                ["No data available yet."],
                self.embeddings,
            )
            logger.info(
                "[VECTOR] No documents found, created placeholder index"
            )
        else:
            chunks = self.text_splitter.split_documents(all_documents)

            logger.info(
                "[VECTOR] Split into %d chunks (chunk_size=%d, overlap=%d)",
                len(chunks),
                CHUNK_SIZE,
                CHUNK_OVERLAP,
            )

            self.vector_store = FAISS.from_documents(
                chunks,
                self.embeddings,
            )

            logger.info(
                "[VECTOR] FAISS index built successfully with %d vectors",
                len(chunks),
            )

        self._retriever_cache.clear()
        self.save_vector_store()

        return self.vector_store

    # ---------------- SAVE VECTOR STORE ---------------- #

    def save_vector_store(self):
        if not self.vector_store:
            return

        try:
            self.vector_store.save_local(str(VECTOR_STORE_DIR))
            logger.info("[VECTOR] Vector store saved to disk")
        except Exception as e:
            logger.error(
                "Failed to save vector store to disk: %s",
                e,
            )

    # ---------------- GET RETRIEVER ---------------- #

    def get_retriever(self, k: int = 10):
        if not self.vector_store:
            # Return a no-op stub retriever when vector store is unavailable
            class _NoOpRetriever:
                def invoke(self, *a, **kw): return []
                def get_relevant_documents(self, *a, **kw): return []
            return _NoOpRetriever()

        if k not in self._retriever_cache:
            self._retriever_cache[k] = self.vector_store.as_retriever(
                search_kwargs={"k": k}
            )

        return self._retriever_cache[k]
