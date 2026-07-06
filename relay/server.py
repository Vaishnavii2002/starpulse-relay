"""
StarPulse Voice Relay Server
Twilio Media Streams + Deepgram STT/TTS + Claude AI (streaming).

Optimizations:
  - Claude 3.5 Haiku + prompt caching for faster TTFT
  - Persistent httpx client for TTS (no TCP+TLS overhead per call)
  - Pipelined REST TTS: Claude keeps generating while earlier chunks synthesize
  - Fast first-chunk trigger (~18 chars / first clause boundary)
  - Auto-hangup via Twilio REST API on terminal responses

Run locally:  uvicorn server:app --host 0.0.0.0 --port 8080
Expose:       ngrok http 8080
"""
import asyncio
import base64
import json
import logging
import os
import re
import time
from typing import Optional

import anthropic
import httpx
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response, HTMLResponse
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("relay")

app = FastAPI(title="StarPulse Voice Relay")

# ── Config ──
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")

SYSTEM_PROMPT = (
    "You are a friendly and professional Medicare care coordinator for StarPulse. "
    "You are speaking with a member on the phone about their healthcare gap. "
    "Keep responses short (1-3 sentences) since this is a voice conversation. "
    "Be warm, empathetic, and helpful. Help them understand why their screening "
    "or medication adherence matters, and assist with scheduling if they want. "
    "If they want to schedule, offer time slots: tomorrow morning 9 AM, "
    "tomorrow afternoon 2 PM, or tomorrow afternoon 3:30 PM. "
    "If they have questions about their health plan, answer helpfully. "
    "Never provide specific medical diagnoses — recommend they speak with their doctor. "
    "End the conversation gracefully when the member is satisfied. "
    "Speak naturally and conversationally. "
    "Avoid lists or bullet points — use flowing sentences. Keep a warm, unhurried tone. "
    "IMPORTANT: Never start your response with filler phrases like 'Sure thing', "
    "'Absolutely', 'Of course', 'Great question', 'Let me help', or 'I would be happy to' — "
    "jump directly into the substantive answer. The system already provides "
    "an acknowledgment phrase before your response plays."
)

GREETING = "Hi, I'm your StarPulse care coordinator. How can I help you today?"

_SENTENCE_END = re.compile(r'[.!?]\s*$')

TERMINAL_PHRASES = {
    "you're welcome", "have a great day", "take care", "goodbye",
    "all set", "see you", "thanks for calling", "have a nice day",
}

# ── Context-aware fillers ──
_FILLER_MAP = {
    "question": "sure thing, let me see.",
    "scheduling": "Let me look into that.",
    "default": "One moment.",
}

_SKIP_KEYWORDS = {
    "ok", "okay", "yes", "yeah", "yep", "sure", "got it", "alright",
    "hi", "hello", "hey", "no", "nah", "nope",
    "thank you", "thanks", "bye", "goodbye", "take care",
}

_QUESTION_KEYWORDS = {"what", "how", "why", "when", "where", "which", "can", "could", "do", "does", "is", "are", "will"}
_SCHEDULING_KEYWORDS = {"schedule", "book", "appointment", "slot", "time", "morning", "afternoon", "available"}


def _pick_filler(user_text: str) -> Optional[str]:
    text = user_text.lower().strip().rstrip(".!?")
    if len(text.split()) <= 3 or text in _SKIP_KEYWORDS:
        return None
    words = set(text.split())
    if words & _SCHEDULING_KEYWORDS:
        return "scheduling"
    if words & _QUESTION_KEYWORDS or text.endswith("?") or user_text.strip().endswith("?"):
        return "question"
    return "default"


# ── Caches & persistent clients ──
_greeting_audio_cache: bytes = b""
_filler_audio_cache: dict[str, bytes] = {}
_claude_client: Optional[anthropic.AsyncAnthropic] = None
_tts_client: Optional[httpx.AsyncClient] = None


def _get_claude():
    global _claude_client
    if _claude_client is None:
        _claude_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    return _claude_client


def _get_tts_client():
    global _tts_client
    if _tts_client is None:
        _tts_client = httpx.AsyncClient(timeout=10.0)
    return _tts_client


# ── Deepgram TTS (persistent client — no per-call handshake) ──
async def deepgram_tts(text: str) -> bytes:
    for attempt in range(2):
        try:
            resp = await _get_tts_client().post(
                "https://api.deepgram.com/v1/speak",
                params={"model": "aura-asteria-en", "encoding": "mulaw", "sample_rate": "8000", "container": "none"},
                headers={"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "application/json"},
                json={"text": text},
            )
            resp.raise_for_status()
            return resp.content
        except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError) as e:
            if attempt == 0:
                logger.warning("TTS connection stale, retrying: %s", e)
                continue
            raise
    return b""


# ── Startup: pre-cache greeting audio ──
@app.on_event("startup")
async def warm_cache():
    global _greeting_audio_cache
    if not DEEPGRAM_API_KEY:
        logger.error("DEEPGRAM_API_KEY not set")
        return
    try:
        _greeting_audio_cache = await deepgram_tts(GREETING)
        logger.info("Cached greeting (%d bytes)", len(_greeting_audio_cache))
        for key, phrase in _FILLER_MAP.items():
            audio = await deepgram_tts(phrase)
            _filler_audio_cache[key] = audio
            logger.info("Cached filler '%s' (%d bytes)", key, len(audio))
    except Exception as e:
        logger.error("Cache warmup failed: %s", e)


async def _keep_alive():
    """Self-ping every 14 min to prevent Render free tier from sleeping."""
    render_url = os.getenv("RENDER_EXTERNAL_URL", "")
    if not render_url:
        return
    await asyncio.sleep(60)
    while True:
        try:
            await _get_tts_client().get(f"{render_url}/health")
            logger.info("Keep-alive ping sent")
        except Exception:
            pass
        await asyncio.sleep(840)

@app.on_event("startup")
async def start_keep_alive():
    asyncio.create_task(_keep_alive())

@app.on_event("shutdown")
async def cleanup():
    if _tts_client:
        await _tts_client.aclose()


# ── Helpers ──
def _is_terminal(text: str) -> bool:
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in TERMINAL_PHRASES)


def _audio_to_frames(stream_sid: str, audio: bytes, chunk_size: int = 640) -> list[str]:
    frames = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i : i + chunk_size]
        frames.append(json.dumps({
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": base64.b64encode(chunk).decode("ascii")},
        }))
    return frames


async def send_audio(ws: WebSocket, stream_sid: str, audio: bytes, mark: Optional[str] = None):
    for frame in _audio_to_frames(stream_sid, audio):
        await ws.send_text(frame)
    if mark:
        await ws.send_text(json.dumps({
            "event": "mark",
            "streamSid": stream_sid,
            "mark": {"name": mark},
        }))


async def clear_audio(ws: WebSocket, stream_sid: str):
    await ws.send_text(json.dumps({"event": "clear", "streamSid": stream_sid}))


async def _hangup_call(call_sid: str):
    """End a Twilio call via REST API."""
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN or not call_sid:
        logger.warning("Cannot hangup: missing Twilio credentials or call_sid")
        return
    url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Calls/{call_sid}.json"
    try:
        resp = await _get_tts_client().post(
            url,
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
            data={"Status": "completed"},
        )
        logger.info("Hangup call %s: status=%d", call_sid, resp.status_code)
    except Exception as e:
        logger.error("Hangup failed for %s: %s", call_sid, e)


# ── Streaming Claude → pipelined REST TTS → Twilio ──
async def stream_claude_and_speak(
    ws: WebSocket, stream_sid: str, conversation: list[dict]
) -> str:
    full_reply = ""
    chunk_num = 0
    t_start = time.time()
    t_first_audio = None
    audio_queue: asyncio.Queue[Optional[tuple[int, bytes]]] = asyncio.Queue()

    async def tts_worker(seq: int, text: str):
        try:
            audio = await deepgram_tts(text)
            await audio_queue.put((seq, audio))
        except Exception as e:
            logger.error("TTS error chunk %d: %s", seq, e)
            await audio_queue.put((seq, b""))

    async def sender():
        nonlocal t_first_audio
        next_seq = 1
        pending: dict[int, bytes] = {}
        while True:
            item = await audio_queue.get()
            if item is None:
                break
            seq, audio = item
            pending[seq] = audio
            while next_seq in pending:
                data = pending.pop(next_seq)
                if data:
                    if t_first_audio is None:
                        await clear_audio(ws, stream_sid)
                        t_first_audio = time.time()
                        logger.info("🔊 FIRST AUDIO: %dms", int((t_first_audio - t_start) * 1000))
                    await send_audio(ws, stream_sid, data)
                next_seq += 1

    sender_task = asyncio.create_task(sender())

    try:
        token_buffer = ""
        last_tts_time = time.time()
        tts_tasks: list[asyncio.Task] = []

        logger.info("🚀 Claude streaming...")

        async with _get_claude().messages.stream(
            model=ANTHROPIC_MODEL,
            max_tokens=200,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=conversation,
        ) as stream:
            async for text in stream.text_stream:
                full_reply += text
                token_buffer += text
                elapsed = int((time.time() - last_tts_time) * 1000)
                buf = token_buffer.rstrip()

                if chunk_num == 0:
                    should_send = (
                        (buf.endswith((',', '.', '!', '?', ';', ':')) and len(buf) >= 8) or
                        (len(token_buffer) >= 18) or
                        (elapsed >= 150 and len(token_buffer) >= 10)
                    )
                else:
                    should_send = (
                        (len(token_buffer) >= 30) or
                        (elapsed >= 300 and len(token_buffer) >= 20) or
                        (buf.endswith(('.', '!', '?')) and len(token_buffer) >= 20)
                    )

                if should_send and token_buffer.strip():
                    chunk_num += 1
                    chunk_text = token_buffer.strip()
                    token_buffer = ""
                    last_tts_time = time.time()
                    tts_tasks.append(asyncio.create_task(tts_worker(chunk_num, chunk_text)))
                    logger.info("Chunk %d: %s...", chunk_num, chunk_text[:40])

            if token_buffer.strip():
                chunk_num += 1
                tts_tasks.append(asyncio.create_task(tts_worker(chunk_num, token_buffer.strip())))

        if tts_tasks:
            await asyncio.gather(*tts_tasks, return_exceptions=True)
        await audio_queue.put(None)
        await sender_task

        await ws.send_text(json.dumps({
            "event": "mark", "streamSid": stream_sid, "mark": {"name": "response"},
        }))

        total_ms = int((time.time() - t_start) * 1000)
        first_ms = int((t_first_audio - t_start) * 1000) if t_first_audio else 0
        logger.info("✓ Done: %dms, 1st=%dms, %d chunks", total_ms, first_ms, chunk_num)

    except Exception as e:
        logger.error("Stream Claude error: %s", e, exc_info=True)
        if not full_reply:
            full_reply = "I apologize, I'm having a brief technical issue. Could you please repeat that?"
            try:
                audio = await deepgram_tts(full_reply)
                await send_audio(ws, stream_sid, audio, mark="response")
            except Exception:
                pass
        await audio_queue.put(None)
        try:
            await sender_task
        except Exception:
            pass

    return full_reply


# ── Deepgram STT via raw WebSocket (additional_headers for auth) ──
async def _connect_deepgram_stt():
    url = (
        "wss://api.deepgram.com/v1/listen?"
        "model=nova-2&language=en-US&encoding=mulaw&sample_rate=8000"
        "&channels=1&punctuate=true&interim_results=true"
        "&endpointing=300&utterance_end_ms=1000"
    )
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
    ws = await websockets.connect(url, additional_headers=headers)
    logger.info("Deepgram STT WebSocket connected")
    return ws


# ── WebSocket handler ──
@app.websocket("/ws/voice-agent")
async def voice_agent(ws: WebSocket):
    await ws.accept()

    call_sid: Optional[str] = None
    stream_sid: Optional[str] = None
    conversation: list[dict] = []
    dg_ws = None
    transcript_queue: asyncio.Queue[str] = asyncio.Queue()
    last_stt_activity = time.time()
    last_speech_detected = time.time()
    first_audio_ts: Optional[float] = None
    STT_WATCHDOG_TIMEOUT = 8.0

    try:
        dg_ws = await _connect_deepgram_stt()

        async def read_deepgram():
            nonlocal last_stt_activity, last_speech_detected
            try:
                async for msg in dg_ws:
                    last_stt_activity = time.time()
                    data = json.loads(msg)
                    if data.get("type") == "Results":
                        channel = data.get("channel", {})
                        alts = channel.get("alternatives", [{}])
                        transcript = alts[0].get("transcript", "")
                        is_final = data.get("is_final", False)
                        speech_final = data.get("speech_final", False)
                        if transcript.strip():
                            last_speech_detected = time.time()
                        if transcript.strip() and is_final and speech_final:
                            logger.info("STT final: %s", transcript.strip())
                            await transcript_queue.put(transcript.strip())
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Deepgram STT connection closed")
            except Exception as e:
                logger.error("Deepgram read error: %s", e)

        dg_task = asyncio.create_task(read_deepgram())

        async def process_transcripts():
            nonlocal conversation
            while True:
                user_text = await transcript_queue.get()
                if user_text is None:
                    break
                if not stream_sid:
                    continue

                t_eos = time.time()

                # ── CONTEXT-AWARE FILLER ──
                filler_key = _pick_filler(user_text)
                if filler_key and filler_key in _filler_audio_cache:
                    await send_audio(ws, stream_sid, _filler_audio_cache[filler_key])
                    logger.info("Filler sent: %s (%s)", _FILLER_MAP[filler_key], filler_key)

                # ── STREAM CLAUDE → PIPELINED TTS → PLAY ──
                conversation.append({"role": "user", "content": user_text})
                ai_reply = await stream_claude_and_speak(ws, stream_sid, conversation)
                conversation.append({"role": "assistant", "content": ai_reply})

                total_ms = int((time.time() - t_eos) * 1000)
                logger.info("Turn complete in %dms: %s", total_ms, ai_reply[:80])

                # ── AUTO-HANGUP on terminal response ──
                if _is_terminal(ai_reply):
                    logger.info("🛑 Terminal response detected. Ending call in 5s...")
                    await asyncio.sleep(5.0)
                    await _hangup_call(call_sid)
                    break

        transcript_task = asyncio.create_task(process_transcripts())

        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            event = msg.get("event", "")

            if event == "connected":
                logger.info("Twilio stream connected")

            elif event == "start":
                start = msg.get("start", {})
                call_sid = start.get("callSid", "")
                stream_sid = start.get("streamSid", "")
                logger.info("Stream started: call=%s stream=%s", call_sid, stream_sid)

                if stream_sid and _greeting_audio_cache:
                    conversation.append({"role": "assistant", "content": GREETING})
                    await send_audio(ws, stream_sid, _greeting_audio_cache, mark="greeting")
                    logger.info("Greeting sent")

            elif event == "media":
                payload = msg.get("media", {}).get("payload", "")
                if payload and dg_ws:
                    if first_audio_ts is None:
                        first_audio_ts = time.time()

                    needs_reconnect = False
                    if dg_task.done():
                        logger.warning("STT reader task died, reconnecting")
                        needs_reconnect = True
                    elif time.time() - last_speech_detected > STT_WATCHDOG_TIMEOUT and first_audio_ts and time.time() - first_audio_ts > STT_WATCHDOG_TIMEOUT:
                        logger.warning("STT watchdog: no speech for %.1fs, reconnecting", time.time() - last_speech_detected)
                        needs_reconnect = True

                    if needs_reconnect:
                        try:
                            await dg_ws.close()
                        except Exception:
                            pass
                        dg_ws = await _connect_deepgram_stt()
                        last_stt_activity = time.time()
                        last_speech_detected = time.time()
                        first_audio_ts = time.time()
                        dg_task = asyncio.create_task(read_deepgram())

                    try:
                        await dg_ws.send(base64.b64decode(payload))
                    except (websockets.exceptions.ConnectionClosed, Exception) as e:
                        logger.warning("STT socket dead, reconnecting: %s", e)
                        try:
                            await dg_ws.close()
                        except Exception:
                            pass
                        dg_ws = await _connect_deepgram_stt()
                        last_stt_activity = time.time()
                        last_speech_detected = time.time()
                        first_audio_ts = time.time()
                        dg_task = asyncio.create_task(read_deepgram())
                        await dg_ws.send(base64.b64decode(payload))

            elif event == "stop":
                logger.info("Stream stopped")
                break

    except WebSocketDisconnect:
        logger.info("Stream disconnected: %s", call_sid)
    except Exception as e:
        logger.error("Relay error: %s", call_sid, exc_info=True)
    finally:
        await transcript_queue.put(None)
        if dg_ws:
            try:
                await dg_ws.close()
                logger.info("Deepgram closed")
            except Exception:
                pass


# ── Twilio DTMF Webhooks ──
_SLOT_MAP = {
    "1": "tomorrow morning at 9 AM",
    "2": "tomorrow afternoon at 2 PM",
    "3": "tomorrow afternoon at 3:30 PM",
    "4": "callback requested",
}


def _twiml(xml_body: str) -> Response:
    return Response(content=f'<?xml version="1.0" encoding="UTF-8"?>{xml_body}', media_type="text/xml")


@app.post("/call-gather")
async def call_gather(request: Request):
    form = await request.form()
    digits = form.get("Digits", "")
    call_sid = form.get("CallSid", "")
    logger.info("call-gather: SID=%s Digits=%s", call_sid, digits)

    if digits == "1":
        return _twiml(
            '<Response>'
            '<Gather numDigits="1" action="/call-slot" method="POST" timeout="10">'
            '<Say voice="Polly.Joanna-Neural">'
            '<prosody rate="95%">'
            'Great! Here are available time slots.'
            '<break time="400ms"/>'
            'Press 1 for tomorrow morning at 9 A M.'
            '<break time="300ms"/>'
            'Press 2 for tomorrow afternoon at 2 P M.'
            '<break time="300ms"/>'
            'Press 3 for tomorrow afternoon at 3 30 P M.'
            '<break time="300ms"/>'
            'Press 4 to request a callback with more options.'
            '</prosody>'
            '</Say>'
            '</Gather>'
            '<Say voice="Polly.Joanna-Neural">We did not receive your selection. A coordinator will call you to schedule. Goodbye.</Say>'
            '</Response>'
        )

    if digits == "2":
        relay_host = os.getenv("RELAY_HOST", "localhost:8080")
        return _twiml(
            '<Response>'
            '<Connect>'
            '<Stream url="wss://' + relay_host + '/ws/voice-agent" />'
            '</Connect>'
            '<Pause length="60"/>'
            '</Response>'
        )

    return _twiml(
        '<Response>'
        '<Say voice="Polly.Joanna-Neural">Thank you for your time. Goodbye.</Say>'
        '</Response>'
    )


@app.post("/call-slot")
async def call_slot(request: Request):
    form = await request.form()
    digits = form.get("Digits", "")
    call_sid = form.get("CallSid", "")
    logger.info("call-slot: SID=%s Digits=%s", call_sid, digits)

    slot = _SLOT_MAP.get(digits, "callback requested")

    if digits == "4":
        return _twiml(
            '<Response>'
            '<Say voice="Polly.Joanna-Neural">Your callback request has been noted. '
            'A care coordinator will reach out to you within 24 hours with more options. '
            'Thank you and have a great day!</Say>'
            '</Response>'
        )

    return _twiml(
        '<Response>'
        '<Say voice="Polly.Joanna-Neural">'
        'You are confirmed for ' + slot + '. '
        'You will receive an SMS confirmation shortly. Thank you and have a great day!</Say>'
        '</Response>'
    )


@app.get("/health")
def health():
    return {"status": "ok", "greeting_cached": len(_greeting_audio_cache) > 0}


@app.get("/")
def root():
    return HTMLResponse("<h1>StarPulse Relay Active</h1>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
