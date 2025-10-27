# Create Clash Royale Emotes using MediaPipe


This project watches your webcam and recognizes a few very specific **face + hand “emotes”**. When an emote is detected and sustained, it can fire an action (play a video, send a hotkey, trigger a script, etc.).  
No training, no cloud — just landmarks + geometry.

Install dependencies with:
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

-----------------------------------------------------------
QUICK START
-----------------------------------------------------------
Run the program:
    source venv/bin/activate  # activate virtual environment
    python main.py

The program will:
- Open your webcam in a "Facial Gesture Detection" window
- Wait for you to stick out your tongue AND shake your head simultaneously
- When detected for 7 frames, it will play a test video in a new window
- Press ESC to quit
---

## Features

- **Runs locally** — no uploads, no servers
- **MediaPipe Face + Hands** landmarks in real time
- Uses only a **small subset of stable landmark relationships**
- Emotes defined using **simple boolean rules**, not ML
- **Sustain + cooldown** logic → no flicker or spam
- Fully **customizable** — edit or replace emotes easily

---

## Example built-in emotes

tbd
