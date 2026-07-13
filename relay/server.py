"""
StarPulse Voice Relay Server
Twilio Media Streams + Deepgram STT/TTS + Claude AI (streaming).

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
# Cheaper/lighter model just for the short post-call summary
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "claude-haiku-4-5")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
# Databricks SQL warehouse — the relay writes call events + summaries straight to
# the audit table (the Databricks App itself is OAuth-walled, so HTTP POST won't work).
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "").rstrip("/")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")
DATABRICKS_WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "")
OUTREACH_TABLE = os.getenv("OUTREACH_TABLE", "medicare_stars.silver.outreach_activity_log")

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
    "IMPORTANT: Never start your response with filler phrases like 'sure thing, let me see.', "
    "'Let me look into that', or 'One moment.'"
    "jump directly into the substantive answer. The system already provides "
    "an acknowledgment phrase before your response plays."
)

GREETING = "Hi, I'm your StarPulse care coordinator. How can I help you today?"

_SENTENCE_END = re.compile(r'[.!?]\s*$')

TERMINAL_PHRASES = {
    "you're welcome", "have a great day", "take care", "goodbye", "see you", "thanks for calling", "have a nice day",
}

# ── Context-aware fillers ──
_FILLER_MAP = {
    "question": "sure thing, let me see.",
    "scheduling": "Let me look",
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
                params={"model": "aura-2-janus-en", "encoding": "mulaw", "sample_rate": "8000", "container": "none"},
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


# ── Report call events + summary straight into the Databricks audit table ──
def _sql_esc(s: str) -> str:
    return (s or "").replace("'", "''")


async def _run_sql(statement: str) -> bool:
    """Execute a SQL statement on the Databricks warehouse via the REST API.
    Returns True only if the statement actually succeeded (a failed statement
    still returns HTTP 200 with state=FAILED, so we must inspect the body)."""
    if not (DATABRICKS_HOST and DATABRICKS_TOKEN and DATABRICKS_WAREHOUSE_ID):
        logger.warning("Databricks SQL not configured — skipping audit write")
        return False
    try:
        # Own timeout > wait_timeout so a cold warehouse spin-up (15-30s) survives.
        resp = await _get_tts_client().post(
            f"{DATABRICKS_HOST}/api/2.0/sql/statements",
            headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
            json={"warehouse_id": DATABRICKS_WAREHOUSE_ID, "statement": statement, "wait_timeout": "30s"},
            timeout=35.0,
        )
        if resp.status_code != 200:
            logger.error("SQL exec HTTP %d: %s", resp.status_code, resp.text[:200])
            return False
        state = resp.json().get("status", {}).get("state", "")
        if state == "SUCCEEDED":
            return True
        if state in ("PENDING", "RUNNING"):  # accepted, still finishing on the warehouse
            logger.info("SQL exec accepted (state=%s)", state)
            return True
        err = resp.json().get("status", {}).get("error", {}).get("message", "")
        logger.error("SQL exec failed (state=%s): %s", state, err)
        return False
    except httpx.TimeoutException:
        logger.error("SQL exec timed out (warehouse slow/cold)")
        return False
    except Exception as e:
        logger.error("SQL exec failed: %s", e)
        return False


async def report_call(call_sid: str, event: str = "", detail: str = "",
                      member_selection: str = "", summary: str = ""):
    """Write an in-call event and/or final summary to the outreach audit table."""
    if not call_sid:
        return
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    ts = time.strftime("%H:%M:%S", time.gmtime())

    if event or member_selection:
        entry = f"[{ts}] {_sql_esc(event or 'in-progress')}"
        if detail:
            entry += f": {_sql_esc(detail)}"
        sets = [
            f"status = '{_sql_esc(event or 'in-progress')}'",
            ("interaction_log = CASE WHEN interaction_log IS NULL OR interaction_log = '' "
             f"THEN '{entry}' ELSE interaction_log || ' | ' || '{entry}' END"),
            f"updated_at = '{now}'",
        ]
        if member_selection:
            sets.append(f"member_selection = '{_sql_esc(member_selection)}'")
        ok = await _run_sql(
            f"UPDATE {OUTREACH_TABLE} SET {', '.join(sets)} "
            f"WHERE provider_sid = '{_sql_esc(call_sid)}'"
        )
        logger.info("Audit event logged [%s]: %s (ok=%s)", call_sid, event or member_selection, ok)

    if summary:
        ok = await _run_sql(
            f"UPDATE {OUTREACH_TABLE} SET call_summary = '{_sql_esc(summary)}', "
            f"updated_at = '{now}' WHERE provider_sid = '{_sql_esc(call_sid)}'"
        )
        logger.info("Audit summary logged [%s] (ok=%s)", call_sid, ok)


async def save_ai_call(call_sid: str, trail: list[str], summary: str) -> bool:
    """Write the full turn-by-turn interaction trail + summary in ONE UPDATE
    (avoids per-turn writes that add latency and race on the same row).
    Retries once so a cold-warehouse timeout doesn't lose the transcript."""
    if not call_sid or not trail:
        return False
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    trail_text = _sql_esc(" | ".join(trail))
    sets = [
        "status = 'ai-conversation'",
        "member_selection = 'AI Agent'",
        ("interaction_log = CASE WHEN interaction_log IS NULL OR interaction_log = '' "
         f"THEN '{trail_text}' ELSE interaction_log || ' | ' || '{trail_text}' END"),
        f"updated_at = '{now}'",
    ]
    if summary:
        sets.append(f"call_summary = '{_sql_esc(summary)}'")
    stmt = (f"UPDATE {OUTREACH_TABLE} SET {', '.join(sets)} "
            f"WHERE provider_sid = '{_sql_esc(call_sid)}'")
    for attempt in range(2):
        if await _run_sql(stmt):
            return True
        logger.warning("save_ai_call retry %d for %s", attempt + 1, call_sid)
    return False


async def summarize_call(conversation: list[dict]) -> str:
    """Use a cheap Claude model to produce one short professional line about the call."""
    if not conversation:
        return ""
    transcript = "\n".join(
        f"{'Member' if m['role'] == 'user' else 'Agent'}: {m['content']}"
        for m in conversation if isinstance(m.get("content"), str)
    )
    prompt = (
        "Summarize this healthcare outreach call in ONE short, professional sentence "
        "for an internal audit log. State the outcome clearly: was an appointment "
        "scheduled and for when, did the member ask questions, or was there no action? "
        "Be concise (max 20 words). Do not add quotes.\n\n"
        f"{transcript}\n\nSummary:"
    )
    try:
        resp = await _get_claude().messages.create(
            model=SUMMARY_MODEL,
            max_tokens=60,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        logger.error("summarize_call failed: %s", e)
        return ""


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
    trail: list[str] = []  # timestamped turn-by-turn log for the audit interaction_log
    dg_ws = None
    transcript_queue: asyncio.Queue[str] = asyncio.Queue()
    greeting_done = False

    dg_task = None
    keepalive_task = None

    try:
        async def read_deepgram():
            # Buffer finalized segments; flush on speech_final OR UtteranceEnd,
            # whichever comes first, so a query is never dropped waiting for a pause.
            buffer_parts: list[str] = []

            async def flush(reason: str):
                if buffer_parts:
                    text = " ".join(buffer_parts).strip()
                    buffer_parts.clear()
                    if text:
                        logger.info("STT final (%s): %s", reason, text)
                        await transcript_queue.put(text)

            try:
                async for msg in dg_ws:
                    data = json.loads(msg)
                    msg_type = data.get("type")
                    if msg_type == "Results":
                        channel = data.get("channel", {})
                        alts = channel.get("alternatives", [{}])
                        transcript = alts[0].get("transcript", "").strip()
                        is_final = data.get("is_final", False)
                        speech_final = data.get("speech_final", False)
                        if transcript and is_final:
                            buffer_parts.append(transcript)
                        if speech_final:
                            await flush("speech_final")
                    elif msg_type == "UtteranceEnd":
                        await flush("utterance_end")
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Deepgram STT connection closed")
            except Exception as e:
                logger.error("Deepgram read error: %s", e)

        async def keepalive_deepgram():
            """Send KeepAlive every 5s so Deepgram never closes the idle connection."""
            while True:
                await asyncio.sleep(5)
                try:
                    await dg_ws.send(json.dumps({"type": "KeepAlive"}))
                except Exception:
                    pass

        async def start_stt():
            """Connect Deepgram fresh and start reader + keepalive tasks."""
            nonlocal dg_ws, dg_task, keepalive_task
            if dg_task:
                dg_task.cancel()
            if keepalive_task:
                keepalive_task.cancel()
            if dg_ws:
                try:
                    await dg_ws.close()
                except Exception:
                    pass
            dg_ws = await _connect_deepgram_stt()
            dg_task = asyncio.create_task(read_deepgram())
            keepalive_task = asyncio.create_task(keepalive_deepgram())

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
                trail.append(f"[{time.strftime('%H:%M:%S', time.gmtime())}] ai-conversation: Member: {user_text}")
                ai_reply = await stream_claude_and_speak(ws, stream_sid, conversation)
                conversation.append({"role": "assistant", "content": ai_reply})
                trail.append(f"[{time.strftime('%H:%M:%S', time.gmtime())}] ai-conversation: AI: {ai_reply}")

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

            elif event == "mark":
                mark_name = msg.get("mark", {}).get("name", "")
                if mark_name == "greeting":
                    # Connect Deepgram FRESH right when the greeting ends — a stale
                    # connection that sat through the greeting won't transcribe reliably.
                    await start_stt()
                    greeting_done = True
                    logger.info("Greeting playback finished — STT connected fresh, now active")

            elif event == "media":
                payload = msg.get("media", {}).get("payload", "")
                if payload and dg_ws and greeting_done:
                    # Reconnect only on a REAL failure (dead reader task or send error)
                    if dg_task.done():
                        logger.warning("STT reader task died, reconnecting")
                        await start_stt()

                    try:
                        await dg_ws.send(base64.b64decode(payload))
                    except (websockets.exceptions.ConnectionClosed, Exception) as e:
                        logger.warning("STT socket dead, reconnecting: %s", e)
                        await start_stt()
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
        try:
            keepalive_task.cancel()
        except Exception:
            pass
        if dg_ws:
            try:
                await dg_ws.close()
                logger.info("Deepgram closed")
            except Exception:
                pass
        # ── Write the full transcript trail + summary to the audit log (one write) ──
        # Only if the member actually spoke (more than just the greeting).
        if call_sid and trail:
            try:
                summary = await summarize_call(conversation)
                if summary:
                    logger.info("Call summary [%s]: %s", call_sid, summary)
                ok = await save_ai_call(call_sid, trail, summary)
                logger.info("AI call saved [%s] (turns=%d, ok=%s)", call_sid, len(trail), ok)
            except Exception as e:
                logger.error("Save AI call failed for %s: %s", call_sid, e)


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
        asyncio.create_task(report_call(call_sid, event="scheduling",
                                        detail="Member pressed 1 — scheduling"))
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
        asyncio.create_task(report_call(call_sid, event="ai-conversation",
                                        detail="Member pressed 2 — AI agent"))
        relay_host = os.getenv("RELAY_HOST", "localhost:8080")
        return _twiml(
            '<Response>'
            '<Connect>'
            '<Stream url="wss://' + relay_host + '/ws/voice-agent" />'
            '</Connect>'
            '<Pause length="60"/>'
            '</Response>'
        )

    asyncio.create_task(report_call(call_sid, event="no-action",
                                    detail=f"Member pressed {digits or 'nothing'}",
                                    member_selection="No action",
                                    summary="No action taken by the member."))
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
        asyncio.create_task(report_call(
            call_sid, event="callback-requested",
            detail="Member pressed 4 — callback", member_selection="No action",
            summary="Member requested a callback with more options."))
        return _twiml(
            '<Response>'
            '<Say voice="Polly.Joanna-Neural">Your callback request has been noted. '
            'A care coordinator will reach out to you within 24 hours with more options. '
            'Thank you and have a great day!</Say>'
            '</Response>'
        )

    asyncio.create_task(report_call(
        call_sid, event="scheduled", detail=f"Slot selected: {slot}",
        member_selection=f"Scheduled — {slot}",
        summary=f"Appointment scheduled for {slot}."))
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
