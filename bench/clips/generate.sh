#!/bin/bash
# Regenerates the bench clips (macOS: uses `say`, `afconvert`, `ffmpeg`,
# `python3`). The WAVs are deliberately NOT committed — ~10 MB of generated
# binaries — so run this once before benching:
#
#   bash bench/clips/generate.sh
#
# Speech clips: one spoken utterance + 14 s of silence, so each loop of
# Chrome's --use-file-for-fake-audio-capture is one conversational turn.
# Noise clips: adversarial inputs for bench/observe.mjs (want ZERO turns).
# Note: `say` output varies by machine/OS voice version — benches compare
# paired conditions on the SAME machine, so this doesn't matter.
set -euo pipefail
cd "$(dirname "$0")"

speech() { # $1 = name, $2 = text
  say -v Samantha -o "/tmp/$1.aiff" "$2"
  afconvert -f WAVE -d LEI16@48000 -c 1 "/tmp/$1.aiff" "/tmp/$1.raw.wav"
  ffmpeg -y -loglevel error -i "/tmp/$1.raw.wav" -af "apad=pad_dur=14" "$1.wav"
  rm -f "/tmp/$1.aiff" "/tmp/$1.raw.wav"
}

speech map_a "What is your favorite hobby to do on the weekend?"
speech map_b "Tell me something interesting about the ocean."
speech holdout_a "How do you usually stay focused when you are working?"

# Quiet speech: same utterance at 0.15x amplitude (AGC / quiet-speaker probe).
ffmpeg -y -loglevel error -i map_b.wav -filter:a "volume=0.15" quiet_speech.wav

# Ambient: 20 s of low-level pink noise (HVAC / room-tone proxy).
ffmpeg -y -loglevel error -f lavfi -i "anoisesrc=color=pink:amplitude=0.08:duration=20" \
  -ar 48000 -ac 1 -sample_fmt s16 noise_ambient.wav

# Typing: irregular 15 ms white-noise clicks at 80-250 ms gaps (keystroke
# transients — the phantom-turn guard's canonical adversary), 20 s.
ffmpeg -y -loglevel error -f lavfi -i "anoisesrc=color=white:amplitude=1:duration=0.015" \
  -ar 48000 -ac 1 -sample_fmt s16 /tmp/click.wav
python3 - <<'EOF'
import wave, struct, random
with wave.open("/tmp/click.wav", "rb") as w:
    click = struct.unpack(f"<{w.getnframes()}h", w.readframes(w.getnframes()))
sr = 48000
out = bytearray()
random.seed(7)  # fixed seed: identical click pattern across regenerations
t = 0.0
while t < 20.0:
    gap = random.uniform(0.08, 0.25)
    out += b"\x00\x00" * int(gap * sr)
    amp = random.uniform(0.4, 1.0)
    out += struct.pack(f"<{len(click)}h", *[int(f * amp) for f in click])
    t += gap + 0.015
with wave.open("noise_typing.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes(bytes(out))
EOF
rm -f /tmp/click.wav

ls -la ./*.wav
echo "clips generated"
