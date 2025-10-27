# Create Clash Royale Emotes using MediaPipe


This project watches your webcam and recognizes a few very specific **face + hand “emotes”**. When an emote is detected and sustained, it can fire an action (play a video, send a hotkey, trigger a script, etc.).  
No training, no cloud — just landmarks + geometry.

To use:
Install mediapipe and opencv-python
Run main.py

-----------------------------------------------------------
QUICK START
-----------------------------------------------------------
Run the program:
    source venv/bin/activate  # activate virtual environment
    python main.py

The program will:
- Open your webcam in a "Facial Gesture Detection" window
- Wait for you to do one of the 4 emotes
- When detected for 7 frames, it will play a test video in a new window
- Press ESC to quit
---

## Built-in Emotes

1. Goblin Crying Emote


https://github.com/user-attachments/assets/d75ff27a-1998-4fff-9789-377ebc695059

2. Knight Mewing Emote
   

https://github.com/user-attachments/assets/128b85c5-d993-4f75-a9d7-ab35005f0233


3. Wizard 67 Emote


https://github.com/user-attachments/assets/e6795044-569e-4c79-bdb4-273b58e7ffc6


   
4. Princess Yawn Emote


https://github.com/user-attachments/assets/9e485524-7a11-47c0-bb5a-21872d3c33a2

