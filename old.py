import cv2
import mediapipe as mp
from collections import deque
import threading
import queue
import time

# === Settings ===
VIDEO_PATH = "test_videos/orca-tongue.mp4"
SHAKE_WINDOW = 15
SHAKE_THRESHOLD = 0.03
TONGUE_THRESHOLD = 0.01
MIN_MOUTH_OPEN = 0.01
TRIGGER_COOLDOWN = 60
SUSTAIN_FRAMES = 7

# === Setup MediaPipe ===
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
cap = cv2.VideoCapture(0)

nose_positions = deque(maxlen=SHAKE_WINDOW)
gesture_frames = 0
cooldown = 0

# === Queue and flag for video frames ===
video_queue = queue.Queue(maxsize=1)
play_video_flag = threading.Event()

# === Function to read video frames in a separate thread ===
def video_reader(video_path):
    while True:
        play_video_flag.wait()  # wait until flagged
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            print("❌ Could not open video:", video_path)
            play_video_flag.clear()
            continue

        while vid.isOpened() and play_video_flag.is_set():
            ret, frame = vid.read()
            if not ret:
                break
            if not video_queue.full():
                video_queue.put(frame)
            time.sleep(0.01)  # reduce CPU usage

        vid.release()
        play_video_flag.clear()  # done playing

# Start video reader thread
threading.Thread(target=video_reader, args=(VIDEO_PATH,), daemon=True).start()

# === Gesture detection functions ===
def detect_tongue(landmarks):
    upper_lip = landmarks[13].y
    lower_lip = landmarks[14].y
    tongue_tip = landmarks[16].y
    mouth_height = lower_lip - upper_lip
    if mouth_height < MIN_MOUTH_OPEN:
        return False
    return (tongue_tip - lower_lip) > TONGUE_THRESHOLD

def detect_head_shake():
    if len(nose_positions) < SHAKE_WINDOW:
        return False
    motion = max(nose_positions) - min(nose_positions)
    return motion > SHAKE_THRESHOLD

# === Main loop ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(frame_rgb)
    gesture_detected = False
    tongue_out = False
    head_shake = False

    if result.multi_face_landmarks:
        face_landmarks = result.multi_face_landmarks[0].landmark
        nose_positions.append(face_landmarks[1].x)

        # Draw facial landmarks
        for landmark in face_landmarks:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Highlight key points
        nose_x = int(face_landmarks[1].x * frame.shape[1])
        nose_y = int(face_landmarks[1].y * frame.shape[0])
        cv2.circle(frame, (nose_x, nose_y), 8, (0, 0, 255), 2)
        
        lip_upper_x = int(face_landmarks[13].x * frame.shape[1])
        lip_upper_y = int(face_landmarks[13].y * frame.shape[0])
        cv2.circle(frame, (lip_upper_x, lip_upper_y), 6, (255, 0, 0), 2)
        
        lip_lower_x = int(face_landmarks[14].x * frame.shape[1])
        lip_lower_y = int(face_landmarks[14].y * frame.shape[0])
        cv2.circle(frame, (lip_lower_x, lip_lower_y), 6, (255, 0, 0), 2)
        
        tongue_x = int(face_landmarks[16].x * frame.shape[1])
        tongue_y = int(face_landmarks[16].y * frame.shape[0])
        cv2.circle(frame, (tongue_x, tongue_y), 6, (0, 255, 255), 2)

        tongue_out = detect_tongue(face_landmarks)
        head_shake = detect_head_shake()

        if tongue_out and head_shake:
            gesture_detected = True

    # Count consecutive frames
    if gesture_detected:
        gesture_frames += 1
    else:
        gesture_frames = 0

    # Trigger video if sustained
    if gesture_frames >= SUSTAIN_FRAMES and cooldown == 0:
        print("Gesture sustained! Starting video...")
        play_video_flag.set()
        cooldown = TRIGGER_COOLDOWN
        gesture_frames = 0

    # Cooldown counter
    if cooldown > 0:
        cooldown -= 1

    # Add status text
    status_y = 30
    cv2.putText(frame, f"Gesture Frames: {gesture_frames}/{SUSTAIN_FRAMES}", (10, status_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    status_y += 30
    
    if tongue_out:
        cv2.putText(frame, "TONGUE DETECTED", (10, status_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        status_y += 30
    else:
        cv2.putText(frame, "No Tongue", (10, status_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        status_y += 30
    
    if head_shake:
        cv2.putText(frame, "HEAD SHAKE DETECTED", (10, status_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        status_y += 30
    else:
        cv2.putText(frame, "No Head Shake", (10, status_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        status_y += 30
    
    if cooldown > 0:
        cv2.putText(frame, f"Cooldown: {cooldown} frames", (10, status_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    
    # Add legend in top-right
    legend_x = frame.shape[1] - 200
    legend_y = 20
    cv2.putText(frame, "Legend:", (legend_x, legend_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    legend_y += 25
    cv2.circle(frame, (legend_x, legend_y), 8, (0, 0, 255), -1)
    cv2.putText(frame, "Nose", (legend_x + 15, legend_y + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    legend_y += 25
    cv2.circle(frame, (legend_x, legend_y), 6, (255, 0, 0), -1)
    cv2.putText(frame, "Lips", (legend_x + 15, legend_y + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    legend_y += 25
    cv2.circle(frame, (legend_x, legend_y), 6, (0, 255, 255), -1)
    cv2.putText(frame, "Tongue", (legend_x + 15, legend_y + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    legend_y += 25
    cv2.circle(frame, (legend_x, legend_y), 2, (0, 255, 0), -1)
    cv2.putText(frame, "All Landmarks", (legend_x + 15, legend_y + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw progress bar
    bar_width = 400
    bar_height = 30
    bar_x = (frame.shape[1] - bar_width) // 2
    bar_y = frame.shape[0] - 50
    
    if SUSTAIN_FRAMES > 0:
        progress = min(gesture_frames / SUSTAIN_FRAMES, 1.0)
        filled_width = int(bar_width * progress)
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), (0, 255, 0), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
    
    # Show webcam
    cv2.imshow("Facial Gesture Detection", frame)

    # Show video if available
    if not video_queue.empty():
        video_frame = video_queue.get()
        cv2.imshow("Video Playback", video_frame)
    elif not play_video_flag.is_set():
        # Video finished, close window
        cv2.destroyWindow("Video Playback")

    if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
