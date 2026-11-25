# Chordy: a music buddy on a Synthesia avatar

Hum it, whistle it, strum it, or share a song. Chordy listens, tells you the key, tempo and chords (or the notes of a melody), shows cards with guitar shapes and piano keys, and can watch you play through your camera.

| Chordy can... | How |
| --- | --- |
| Hear humming, whistling, an instrument or a song on your speakers | Keeps the last 45 s of your mic and analyses it with librosa (`music.py`): key, tempo, chords or melody notes |
| Name the song | Optional: AudD song recognition, with `AUDD_API_TOKEN` |
| Read a file you upload | The page decodes it, sends 45 s of it to Chordy, which analyses it the same way |
| Look at a link | YouTube, Spotify and SoundCloud links give the title and artist (it can't hear a link); a direct MP3/WAV link is downloaded and analysed |
| Teach a chord | A card with the guitar shape and the piano keys |
| Watch you play | Turn on your camera and ask; Chordy looks at a frame with a vision model |

Chordy is honest about its senses: speech recognition only hears words, so it uses its tools to actually hear music, and says when an answer is an estimate or from memory. It won't recite song lyrics; it talks about them and links to the song page.

## Run it

Same accounts as Practice Partner: LiveKit Cloud and a Synthesia interactive avatar.

```bash
cd ~/Documents/chordy
cp ../interview-coach/.env .env    # or ../turn-taking-lab/.env
uv sync
```

Two Terminal tabs:

```bash
uv run agent.py dev     # tab 1: Chordy
uv run server.py        # tab 2: the page, at http://localhost:8080
```

Press **Start a session**, then **Listen to me play** (or just ask "Chordy, what key is this?"). Headphones help when you're talking; for music on your speakers, Chordy switches the mic to **Music mic** while it listens, so the voice filters don't flatten the music.

**No keys?** `http://localhost:8080/?demo` shows the whole page with scripted cards.

**Song recognition** (optional): get an API token at audd.io and add `AUDD_API_TOKEN=...` to `.env`. Without it, Chordy still finds key, tempo and chords, but can't name a recording.

## Tips for good results

- **A song on speakers:** turn it up, keep it playing for the whole listen, and stay quiet.
- **Humming or whistling:** one clear note at a time, close to the mic.
- **The first listen after starting** can take a few seconds longer while the analysis warms up.
- It's an estimate: fast songs, heavy distortion or busy mixes can confuse the chords. The key is usually the most reliable.

## Put it online

Exactly like Practice Partner (see that project's online guide): the agent goes to LiveKit Cloud with `lk agent create --secrets-file secrets.env` from this folder, and the page goes to Render with the same settings. Chordy's agent is named `chordy`, so it can run in the same LiveKit project as the interview coach.
