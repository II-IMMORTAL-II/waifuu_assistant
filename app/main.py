import json
import logging
import time
import base64
import asyncio
import re
import sys
import threading
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

import uvicorn
import edge_tts

from app.models import ChatRequest, ChatResponse, TTSRequest
from app.services.vector_store import VectorStoreService
from app.services.groq_service import GroqService, AllGroqApisFailedError
from app.services.realtime_service import RealtimeGroqService
from app.services.chat_service import ChatService
from config import *

# ==========================================================
# NEON TERMINAL COLOR
# ==========================================================

GREEN = "\033[92m"
CYAN = "\033[96m"
RESET = "\033[0m"

# ==========================================================
# LOGGING
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("J.A.R.V.I.S")

# ==========================================================
# CINEMATIC BOOT
# ==========================================================

def boot_progress(label):
    for i in range(1, 21):
        bar = "█" * i + "-" * (20 - i)
        sys.stdout.write(f"\r{GREEN}[{bar}] {label}{RESET}")
        sys.stdout.flush()
        time.sleep(0.03)
    print()

def print_title():
    print(GREEN + r"""
╔══════════════════════════════════════════════════════════════╗
║        ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗              ║
║        ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝              ║
║        ██║███████║██████╔╝██║   ██║██║███████╗              ║
║   ██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║              ║
║   ╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║              ║
║    ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝              ║
║         Just A Rather Very Intelligent System                ║
╚══════════════════════════════════════════════════════════════╝
""" + RESET)

# ==========================================================
# GLOBAL SERVICES
# ==========================================================

vector_store_service = None
groq_service = None
realtime_service = None
chat_service = None

# ==========================================================
# LIFESPAN
# ==========================================================

def _init_services():
    """Run in a background thread — keeps event loop free so health check responds immediately."""
    global vector_store_service, groq_service, realtime_service, chat_service
    try:
        logger.info("[BOOT] Initializing Vector Store...")
        vector_store_service = VectorStoreService()
        vector_store_service.create_vector_store()

        logger.info("[BOOT] Connecting Groq AI...")
        groq_service = GroqService(vector_store_service)

        logger.info("[BOOT] Connecting Realtime Engine...")
        realtime_service = RealtimeGroqService(vector_store_service)

        logger.info("[BOOT] Launching Chat Core...")
        chat_service = ChatService(groq_service, realtime_service)

        print(CYAN + "\nJ.A.R.V.I.S ONLINE\n" + RESET)
    except Exception as e:
        logger.error("[BOOT] Service initialization failed: %s", e)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print_title()
    # Initialize services synchronously so health check succeeds but chat is ready
    _init_services()
    yield

    logger.info("Saving sessions...")
    if chat_service:
        for sid in list(chat_service.sessions.keys()):
            chat_service.save_chat_session(sid)

# ==========================================================
# FASTAPI INIT
# ==========================================================

app = FastAPI(title="J.A.R.V.I.S ULTIMATE", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# REQUEST LOGGER
# ==========================================================

class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        logger.info(
            "%s %s (%.3fs)",
            request.method,
            request.url.path,
            time.perf_counter() - t0,
        )
        return response

app.add_middleware(TimingMiddleware)

# ==========================================================
# SENTENCE SPLITTER
# ==========================================================

sentence_regex = re.compile(r'(?<=[.!?]) +')

def split_sentences(text: str):
    return sentence_regex.split(text)

# ==========================================================
# TTS
# ==========================================================

tts_pool = ThreadPoolExecutor(max_workers=4)

def generate_tts_sync(text: str) -> bytes:
    async def _inner():
        communicate = edge_tts.Communicate(
            text=text,
            voice=TTS_VOICE,
            rate=TTS_RATE,
        )
        audio = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio += chunk["data"]
        return audio
    return asyncio.run(_inner())

# ==========================================================
# STREAM ENGINE
# ==========================================================

async def stream_response_generator(session_id, chunk_iter, tts_enabled=False):

    yield f"data: {json.dumps({'session_id': session_id})}\n\n"

    buffer = ""

    for chunk in chunk_iter:

        yield f"data: {json.dumps({'chunk': chunk})}\n\n"

        if tts_enabled:
            buffer += chunk
            sentences = split_sentences(buffer)

            for sentence in sentences[:-1]:
                future = tts_pool.submit(generate_tts_sync, sentence)
                audio = future.result()
                b64 = base64.b64encode(audio).decode("ascii")
                yield f"data: {json.dumps({'audio': b64})}\n\n"

            buffer = sentences[-1]

    yield f"data: {json.dumps({'done': True})}\n\n"

# ==========================================================
# ROUTES
# ==========================================================

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    from fastapi.responses import Response
    return Response(content=b"", media_type="image/x-icon")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = chat_service.get_or_create_session(request.session_id)
    try:
        response = chat_service.process_message(session_id, request.message)
        return ChatResponse(response=response, session_id=session_id)
    except AllGroqApisFailedError as e:
        raise HTTPException(503, str(e))

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    session_id = chat_service.get_or_create_session(request.session_id)
    chunk_iter = chat_service.process_message_stream(
        session_id,
        request.message,
    )
    return StreamingResponse(
        stream_response_generator(session_id, chunk_iter, request.tts),
        media_type="text/event-stream",
    )

@app.post("/chat/realtime/stream")
async def chat_realtime_stream(request: ChatRequest):
    session_id = chat_service.get_or_create_session(request.session_id)
    chunk_iter = chat_service.process_realtime_stream(
        session_id,
        request.message,
    )
    return StreamingResponse(
        stream_response_generator(session_id, chunk_iter, request.tts),
        media_type="text/event-stream",
    )

@app.post("/tts")
async def tts(request: TTSRequest):

    async def generate():
        communicate = edge_tts.Communicate(
            text=request.text,
            voice=TTS_VOICE,
            rate=TTS_RATE,
        )
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    return StreamingResponse(generate(), media_type="audio/mpeg")

# ==========================================================
# FRONTEND
# ==========================================================

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"

if frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=str(frontend_dir), html=True))

@app.get("/")
async def root():
    return RedirectResponse("/app/")

# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    uvicorn.run("app.main:app",
                host="0.0.0.0",
                port=8000,
                reload=True)
