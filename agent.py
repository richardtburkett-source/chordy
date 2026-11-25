"""
Chordy: a music buddy on a Synthesia interactive avatar
=======================================================

Chordy listens to music (humming, whistling, an instrument, a song on your
speakers, a file or a link), tells you what's in it (key, tempo, chords,
notes, and who it is if a song-recognition key is set), shows cards with
the details, and can look at you playing through your camera.

    uv run agent.py dev        # then, in a second window: uv run server.py
    uv run agent.py console    # voice-only quick test in Terminal

How it hears music: speech recognition only understands words, so Chordy
also keeps the last 45 seconds of your microphone audio and analyses it
with librosa when you ask (music.py). Song identification uses AudD when
AUDD_API_TOKEN is set (lookup.py).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import deque

import numpy as np
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, RunContext, cli, function_tool, inference, llm

load_dotenv(".env")

import lookup  # noqa: E402  (reads AUDD_API_TOKEN after .env is loaded)
import music  # noqa: E402

logger = logging.getLogger("chordy")

AGENT_NAME = "chordy"
TOPIC = "chordy"            # data messages to and from the web page
FILE_TOPIC = "music-file"   # audio files sent from the page
BUFFER_SECONDS = 45

STT_MODEL = os.getenv("STT_MODEL", "deepgram/nova-3")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4.1-mini")
VISION_MODEL = os.getenv("VISION_MODEL", LLM_MODEL)
TTS_MODEL = os.getenv("TTS_MODEL", "cartesia/sonic-3")
AVATAR_ID = (os.getenv("CHORDY_AVATAR_ID") or os.getenv("SYNTHESIA_AVATAR_ID") or "").strip()

INSTRUCTIONS = """
You are Chordy, a laid-back, encouraging music teacher on a live video call. You love helping people
understand and play music. Keep every turn short: one to three sentences of plain spoken English, no
lists or markdown. Use musical terms, but explain them simply if the person seems new.

What you can and can't perceive (be honest about this, always):
- Your speech recognition hears WORDS only. It can't hear music. Music, humming or whistling may show up
  as garbled or odd words: if a transcript looks like that, don't answer its content; say it sounds like
  music is playing and offer to listen properly.
- To actually hear music, use the tools. Before calling `listen`, say a short cue like "Okay, go for it,
  I'm listening." Use `analyze_recent` when they ask about something they just played.
- If they ask "can you hear this / my music / me playing?" or say they're about to play, don't explain
  your limits first: say the cue and call `listen` straight away.
- If a tool says it heard nothing or very little, say so plainly, give the level it reported, and suggest
  moving the instrument or speaker closer to the mic. Don't guess what was played.
- You can only see the person when their camera is on, and only through `look_at_camera`.
  Never claim to have heard or seen something you didn't analyse with a tool.
- You can't open links or hear a song from a YouTube or Spotify link, only read its title. To hear a song,
  they play it on speakers near the mic, or share an audio file.
- When they say they're playing a chord ("here's my G"), or ask if it's right, call `check_my_chord` with
  that chord. It listens and compares the notes it hears with the chord they named. Believe its verdict
  over their word: if it says notes are missing, tell them which string or note, kindly. A single camera
  frame can only show rough hand position, never which notes ring, so use `look_at_camera` as a second
  opinion and say that it's rough.
- The analysis is an estimate. Say "sounds like" and mention when it's unsure. A hummed melody gives you
  note names, so you may guess the song from them, but call it a guess.

Cards: the person's screen shows cards next to you. Results from `listen`, `analyze_recent` and shared files
or links appear as cards automatically. Use `show_chord` whenever you name a chord they should play, and
`show_card` for song facts or short practice steps. Then talk about the card briefly; don't read it out.

Songs: you may share what you know about a song, artist or style, but say when it's from memory rather than
measured. Never sing or recite more than a few words of song lyrics; talk about what the lyrics are about
instead, and point them to the lyrics link if there is one.

Teaching: when someone wants to learn a song or chord, show the chord, give one clear instruction at a time,
then offer to watch them play with the camera or listen to them play.
""".strip()

NOTE_INDEX = {n: i for i, n in enumerate(music.NOTES)}
NOTE_INDEX.update({"Db": 1, "D#": 3, "Gb": 6, "G#": 8, "A#": 10, "Cb": 11, "Fb": 4, "E#": 5, "B#": 0})
QUALITIES = {
    "": (0, 4, 7), "maj": (0, 4, 7), "m": (0, 3, 7), "min": (0, 3, 7), "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11), "m7": (0, 3, 7, 10), "sus2": (0, 2, 7), "sus4": (0, 5, 7),
    "dim": (0, 3, 6), "aug": (0, 4, 8), "6": (0, 4, 7, 9), "m6": (0, 3, 7, 9), "9": (0, 4, 7, 10, 14),
    "add9": (0, 4, 7, 14),
}


def chord_notes(name: str) -> tuple[str, list[str]] | None:
    m = re.fullmatch(r"\s*([A-Ga-g])([#b]?)\s*(maj7|maj|min|m7|m6|m|7|sus2|sus4|dim|aug|6|9|add9)?\s*", name)
    if not m:
        return None
    root = m.group(1).upper() + m.group(2)
    quality = m.group(3) or ""
    r = NOTE_INDEX[root]
    label = root + ("m" if quality == "min" else "" if quality == "maj" else quality)
    sharps = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    flats = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
    names = sharps if ("#" in root or root in ("G", "D", "A", "E", "B")) else flats if ("b" in root or root == "F") else music.NOTES
    return label, [names[(r + i) % 12] for i in QUALITIES[quality]]


# --------------------------------------------------------------------------
# What Chordy hears and sees, outside the speech pipeline
# --------------------------------------------------------------------------
class Ears:
    """The last BUFFER_SECONDS of the person's microphone, as raw audio."""

    def __init__(self) -> None:
        self.frames: deque[tuple[float, np.ndarray]] = deque()
        self.task: asyncio.Task | None = None
        self.count = 0
        self.on_music_track = False

    def attach(self, track: rtc.Track, music_track: bool = False) -> None:
        if self.on_music_track and not music_track:
            return   # keep listening on the unfiltered track; the voice mic is only a fallback
        if self.task:
            self.task.cancel()
        self.on_music_track = music_track
        print(f"    [ears] attached to the {'unfiltered music' if music_track else 'voice'} microphone track {track.sid}")
        self.task = asyncio.create_task(self._read(track))
        self.task.add_done_callback(self._done)

    @staticmethod
    def _done(task: asyncio.Task) -> None:
        # a background task that dies quietly would leave Chordy deaf with no message
        if not task.cancelled() and task.exception():
            logger.error("microphone reader stopped: %r", task.exception())

    async def _read(self, track: rtc.Track) -> None:
        stream = rtc.AudioStream(track, sample_rate=music.SR, num_channels=1)
        async for ev in stream:
            now = time.time()
            self.count += 1
            if self.count == 1:
                print("    [ears] first microphone audio is arriving")
            self.frames.append((now, np.frombuffer(ev.frame.data, dtype=np.int16).copy()))
            while self.frames and now - self.frames[0][0] > BUFFER_SECONDS:
                self.frames.popleft()

    def since(self, t0: float) -> np.ndarray:
        parts = [f for t, f in self.frames if t >= t0]
        return (np.concatenate(parts).astype(np.float32) / 32768.0) if parts else np.zeros(0, np.float32)

    def last(self, seconds: float) -> np.ndarray:
        return self.since(time.time() - seconds)


class Eyes:
    """The latest frame from the person's camera, if it's on."""

    def __init__(self) -> None:
        self.frame: rtc.VideoFrame | None = None
        self.at = 0.0
        self.task: asyncio.Task | None = None

    def attach(self, track: rtc.Track) -> None:
        if self.task:
            self.task.cancel()
        self.task = asyncio.create_task(self._read(track))

    async def _read(self, track: rtc.Track) -> None:
        async for ev in rtc.VideoStream(track):
            self.frame, self.at = ev.frame, time.time()

    def detach(self) -> None:
        if self.task:
            self.task.cancel()
        self.frame, self.task = None, None


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------
class Chordy(Agent):
    def __init__(self, ears: Ears, eyes: Eyes, send, vision_llm: llm.LLM, console: bool) -> None:
        super().__init__(instructions=INSTRUCTIONS)
        self.ears, self.eyes, self.send, self.vision_llm, self.console = ears, eyes, send, vision_llm, console
        self.listening = False
        self.mic_ready = asyncio.Event()   # the page sets this once "Music mic" is on

    async def _record(self, context: RunContext, seconds: int) -> np.ndarray:
        """Wait for my cue to finish, switch the page to Music mic, then collect `seconds` of raw mic audio."""
        await context.wait_for_playout()        # don't record my own cue
        await asyncio.sleep(0.4)
        if self.listening:                      # a second listen at the same time would cut the first one short
            return np.zeros(0, np.float32)
        self.listening = True
        self.mic_ready.clear()
        context.session.input.set_audio_enabled(False)   # music isn't a turn to answer
        await self.send({"type": "listening", "seconds": seconds})
        try:
            await asyncio.wait_for(self.mic_ready.wait(), timeout=2.5)   # filters are off before we record
        except asyncio.TimeoutError:
            logger.warning("the page didn't confirm Music mic; recording anyway")
        t0 = time.time()
        try:
            await asyncio.sleep(seconds)
            clip = self.ears.since(t0)
        finally:
            context.session.input.set_audio_enabled(True)
            self.listening = False
            await self.send({"type": "listening_done"})
        peak = float(np.max(np.abs(clip))) if clip.size else 0.0
        level = 0.0
        if clip.size > 2048:   # typical level (90th percentile of short-window RMS), not just the loudest click
            win = clip[: clip.size // 1024 * 1024].reshape(-1, 1024)
            level = float(np.percentile(np.sqrt((win**2).mean(axis=1)), 90))
        print(f"    [ears] recorded {clip.size / music.SR:.1f} s, peak {peak:.4f}, typical level {level:.4f}, mic frames so far {self.ears.count}")
        self._save_clip(clip)
        return clip

    @staticmethod
    def _save_clip(clip: np.ndarray) -> None:
        """Keep the last few recordings in ./debug_clips so a bad result can be checked against the real audio."""
        if not clip.size:
            return
        try:
            folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_clips")
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, time.strftime("clip-%H%M%S.wav"))
            with open(path, "wb") as f:
                f.write(music.to_wav_bytes(clip))
            for old in sorted(os.listdir(folder))[:-8]:
                os.remove(os.path.join(folder, old))
            print(f"    [ears] saved {path}")
        except Exception as e:
            logger.warning("couldn't save debug clip: %s", e)

    async def _analyse(self, y: np.ndarray, source: str) -> dict:
        result = await asyncio.to_thread(music.analyze, y, source=source)
        result.pop("_pitch_classes_per_frame", None)
        card = {k: v for k, v in result.items() if k not in ("notes", "kind")}
        card.update(kind="analysis", sound=result.get("kind", ""), notes=[n["note"] for n in result.get("notes", [])][:32])
        await self.send({"type": "card", "card": card})
        if result.get("heard") and lookup.AUDD_TOKEN and "full music" in result.get("kind", ""):
            try:
                song = await lookup.identify_audio(music.to_wav_bytes(y[-12 * music.SR:]))
            except Exception as e:
                logger.warning("song recognition failed: %s", e)
                song = None
            if song:
                await self.send({"type": "card", "card": song})
                result["identified_song"] = {k: song[k] for k in ("title", "artist", "album", "year")}
            else:
                result["identified_song"] = "no match"
        elif not lookup.AUDD_TOKEN:
            result["identified_song"] = "song recognition isn't switched on, so you can't name recordings from audio"
        result.pop("notes", None)
        return result

    @function_tool
    async def listen(self, context: RunContext, seconds: int = 10) -> str:
        """Listen to music the person is about to play, hum, whistle, sing or play on their speakers,
        then analyse it (key, tempo, chords or melody notes, and the song if recognition is on).
        Say a short cue first, like "Okay, I'm listening."

        Args:
            seconds: how long to listen, 5 to 20. Use 10 unless they ask for longer.
        """
        if self.console:
            return "Listening for music only works in the web page, not in console mode."
        seconds = max(5, min(20, int(seconds or 10)))
        if self.listening:
            return "Already listening right now. Don't start another one; wait for the result."
        clip = await self._record(context, seconds)
        if clip.size == 0:
            return "No microphone audio reached me. Ask them to check the mic is on and try again."
        return json.dumps(await self._analyse(clip, "microphone"))

    @function_tool
    async def check_my_chord(self, context: RunContext, chord: str, seconds: int = 6) -> str:
        """Check by ear whether the person is really playing the chord they say they are. Use it whenever they
        say "I'm playing G" or ask "is this right?" on guitar or piano. Say a short cue first, like
        "Okay, strum it a few times, I'm listening."

        Args:
            chord: the chord they are trying to play, like "G", "Em", "C7".
            seconds: how long to listen, 4 to 10.
        """
        parsed = chord_notes(chord)
        if not parsed:
            return f"'{chord}' isn't a chord name I can check. Ask for something like G, Em, C7, Fmaj7."
        if self.console:
            return "Checking a chord by ear only works in the web page."
        label, notes = parsed
        clip = await self._record(context, max(4, min(10, int(seconds or 6))))
        if clip.size == 0:
            return "No microphone audio reached me. Ask them to check the mic is on and try again."
        result = await asyncio.to_thread(music.check_chord, clip, label, notes)
        await self.send({"type": "card", "card": {"kind": "info", "title": f"Checking {label}", "lines": [result["summary"]]}})
        return json.dumps(result)

    @function_tool
    async def analyze_recent(self, context: RunContext, seconds: int = 15) -> str:
        """Analyse music the person ALREADY played in the last few seconds (up to 40),
        when they ask "what key was that?" or similar.

        Args:
            seconds: how far back to analyse, 5 to 40.
        """
        if self.console:
            return "Analysing recent audio only works in the web page."
        seconds = max(5, min(40, int(seconds or 15)))
        return json.dumps(await self._analyse(self.ears.last(seconds), "microphone, just played"))

    @function_tool
    async def show_chord(self, context: RunContext, chord: str) -> str:
        """Show a chord card with a guitar diagram and piano keys.

        Args:
            chord: chord name like "G", "Em", "C7", "Fmaj7", "Dsus4".
        """
        parsed = chord_notes(chord)
        if not parsed:
            return f"'{chord}' isn't a chord name I can draw. Use names like G, Em, C7, Fmaj7, Dsus4."
        label, notes = parsed
        await self.send({"type": "card", "card": {"kind": "chord", "name": label, "notes": notes}})
        return f"Showing {label}: notes {' '.join(notes)}."

    @function_tool
    async def show_card(self, context: RunContext, title: str, lines: list[str]) -> str:
        """Show a card of facts or practice steps on the person's screen.

        Args:
            title: short title, e.g. "Practice: the G to C change".
            lines: 2 to 6 short lines.
        """
        await self.send({"type": "card", "card": {"kind": "info", "title": title[:80], "lines": [l[:160] for l in lines[:6]]}})
        return "Card shown."

    @function_tool
    async def look_at_camera(self, context: RunContext, question: str) -> str:
        """Look at the person through their camera, e.g. to check their hand position on guitar or piano.

        Args:
            question: what to look for, e.g. "Is their left hand making a G chord correctly?"
        """
        if not self.eyes.frame or time.time() - self.eyes.at > 5:
            return "Their camera is off, so you can't see them. Ask them to switch it on with the camera button."
        await self.send({"type": "looking"})
        ctx = llm.ChatContext.empty()
        ctx.add_message(role="system", content=(
            "You are a music teacher looking at a single camera frame of a student. Describe only what is "
            "clearly visible that helps answer the question: instrument, hand and finger position, posture. "
            "If it's not visible or unclear, say so. Two or three sentences."))
        ctx.add_message(role="user", content=[question, llm.ImageContent(image=self.eyes.frame, inference_width=768, inference_height=768)])
        try:
            async with self.vision_llm.chat(chat_ctx=ctx) as stream:
                res = await stream.collect()
            return res.text.strip() or "The image was unclear."
        except Exception as e:
            logger.exception("vision failed")
            return f"Couldn't look right now ({type(e).__name__})."


server = AgentServer()


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    console = ctx.is_fake_job()
    ears, eyes = Ears(), Eyes()

    async def send(msg: dict) -> None:
        if console:
            print("    [card]", json.dumps(msg)[:200])
            return
        try:
            await ctx.room.local_participant.publish_data(json.dumps(msg), reliable=True, topic=TOPIC)
        except Exception:
            logger.exception("couldn't send %s", msg.get("type"))

    # warm up librosa (its first run compiles code and takes ~15 s)
    asyncio.create_task(asyncio.to_thread(music.analyze, np.random.default_rng(0).normal(0, 0.1, music.SR * 3).astype(np.float32)))

    session = AgentSession(
        stt=STT_MODEL,
        llm=LLM_MODEL,
        tts=TTS_MODEL,
        vad=inference.VAD(),
        turn_handling={
            "turn_detection": inference.TurnDetector(),
            "endpointing": {"mode": "fixed", "min_delay": 0.4, "max_delay": 2.5},
            "interruption": {"enabled": True, "min_duration": 0.6},
        },
    )
    agent = Chordy(ears, eyes, send, inference.LLM(model=VISION_MODEL), console)

    # ---- the person's raw mic and camera ---------------------------------
    def is_person(participant: rtc.RemoteParticipant) -> bool:
        # skip the avatar (and any other agent): we only want the person's mic and camera
        return participant.kind != rtc.ParticipantKind.PARTICIPANT_KIND_AGENT and not participant.identity.startswith("synthesia")

    def _on_track(track: rtc.Track, pub: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant) -> None:
        if not is_person(participant):
            return
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            ears.attach(track, music_track=(pub.name == "music-mic"))
        elif track.kind == rtc.TrackKind.KIND_VIDEO:
            eyes.attach(track)
            asyncio.create_task(send({"type": "camera", "on": True}))

    def _on_untrack(track: rtc.Track, pub, participant) -> None:
        if track.kind == rtc.TrackKind.KIND_VIDEO and is_person(participant):
            eyes.detach()
            asyncio.create_task(send({"type": "camera", "on": False}))

    # ---- songs shared from the page ----------------------------------------
    file_info: dict = {}   # how long the shared song really is, sent by the page just before the excerpt

    async def share_reply(text: str) -> None:
        if agent.listening:
            return
        session.interrupt()
        session.generate_reply(instructions=text)

    async def handle_file(reader: rtc.ByteStreamReader) -> None:
        name = reader.info.name or "a file"
        data = bytearray()
        async for chunk in reader:
            data.extend(chunk)
        try:
            y = await asyncio.to_thread(music.load_audio, bytes(data))
            result = await agent._analyse(y, f"shared file: {name}")
            info = file_info.get("latest") or {}
            scope = ""
            if info.get("total"):
                scope = (f" The whole song is {info['total'] // 60}:{info['total'] % 60:02d} long, but you only analysed an excerpt of "
                         f"{info['length']} seconds starting at {info['start']} s, so the key, tempo and chords describe that excerpt, "
                         "not necessarily the whole song. Say so.")
            await share_reply(f"The person shared an audio file called '{name}'. Your analysis (JSON): "
                              f"{json.dumps(result)}.{scope} In two sentences, tell them what you hear, then ask what they'd like to know or learn.")
        except Exception as e:
            logger.exception("file analysis failed")
            await share_reply(f"The person shared a file but you couldn't read it ({type(e).__name__}). Say so briefly and suggest an MP3 or WAV.")

    async def handle_link(url: str) -> None:
        try:
            got = await lookup.describe_link(url)
        except Exception as e:
            await share_reply(f"The person shared a link ({url}) but it couldn't be opened: {e}. Tell them briefly.")
            return
        if "card" in got:
            card = got["card"]
            await send({"type": "card", "card": card})
            await share_reply(
                f"The person shared a link: {card.get('title')} by {card.get('artist')} ({card.get('source')}). You can't hear "
                "the audio from a link, only see its title. Share what you know about the song from memory (say it's from "
                "memory, not measured), and offer to listen if they play it on their speakers.")
        else:
            y = await asyncio.to_thread(music.load_audio, got["audio"])
            result = await agent._analyse(y, f"shared link: {got['name']}")
            await share_reply(f"The person shared an audio file link. Your analysis (JSON): {json.dumps(result)}. "
                              "In two sentences, tell them what you hear and ask what they'd like to know.")

    def _on_data(packet: rtc.DataPacket) -> None:
        if getattr(packet, "topic", "") != TOPIC:
            return
        try:
            msg = json.loads(packet.data.decode())
        except Exception:
            return
        if msg.get("type") == "link" and isinstance(msg.get("url"), str):
            asyncio.create_task(handle_link(msg["url"].strip()[:500]))
        elif msg.get("type") == "file_info":
            file_info["latest"] = {k: msg.get(k) for k in ("name", "total", "start", "length")}
        elif msg.get("type") == "mic_ready":
            agent.mic_ready.set()
        elif msg.get("type") == "listen":
            session.interrupt()
            session.generate_reply(instructions="The person pressed the Listen button. Say a short cue like "
                                   "'Go for it, I'm listening' and call the listen tool for 10 seconds.")

    if not console:
        ctx.room.on("track_subscribed", _on_track)
        ctx.room.on("track_unsubscribed", _on_untrack)
        ctx.room.on("data_received", _on_data)
        ctx.room.register_byte_stream_handler(FILE_TOPIC, lambda reader, who: asyncio.create_task(handle_file(reader)))

    # ---- the face ------------------------------------------------------------
    if AVATAR_ID and not console:
        from livekit.plugins import synthesia

        print(f"    Starting Synthesia avatar {AVATAR_ID} ...")
        try:
            avatar = synthesia.AvatarSession(synthesia.AvatarConfig(avatar_ids=[AVATAR_ID]))
            await avatar.start(session, room=ctx.room)
        except Exception as e:
            print(f"\n!! The Synthesia avatar failed to start: {type(e).__name__}: {e}\n"
                  "   Check SYNTHESIA_API_KEY and the interactive avatar ID in .env.\n")
            await send({"type": "error", "message": f"The avatar failed to start: {e}"})
            raise

    await session.start(agent=agent, room=ctx.room)
    # pick up tracks published before we started listening for them
    if not console:
        for p in ctx.room.remote_participants.values():
            for pub in p.track_publications.values():
                if pub.track:
                    _on_track(pub.track, pub, p)
    print(f"\n=== Chordy is live | song recognition {'ON' if lookup.AUDD_TOKEN else 'off (set AUDD_API_TOKEN)'} ===\n")
    await session.generate_reply(
        instructions="Say hi as Chordy in one or two short sentences: you can listen to anything they play, hum or "
        "whistle, or a song they share, and tell them what's going on in it. Ask what they've got for you."
    )


if __name__ == "__main__":
    cli.run_app(server)
