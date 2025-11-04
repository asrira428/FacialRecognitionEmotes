"""
Facial & Hand Gesture Emote Detection System
============================================

Detects 5 custom emotes and plays corresponding videos:

Emote 1 - Both Hands Near Eyes: Both hands must have fingertips near eyes → emote1.mp4
Emote 2 - Finger Reaches Jawline: Any fingertip near the jawline area → emote2.mp4
Emote 3 - Hands Up/Down Movement: Both hands move vertically more than threshold while in camera → emote3.mp4
Emote 4 - Mouth Open + Hand Near Face: Mouth opens + hand within threshold distance to face → emote4.mp4
Emote 5 - Booty Shake: MediaPipe Pose detects hip reversal (left hip appears right of right hip) → emote5.mp4

Press ESC to exit.
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import threading
import queue
import time

# === Settings ===
VIDEO_PATH_1 = "emote_videos/emote1.mp4"  # Both hands near eyes
VIDEO_PATH_2 = "emote_videos/emote2.mp4"  # Finger reaches jawline
VIDEO_PATH_3 = "emote_videos/emote3.mp4"  # Hands up/down movement
VIDEO_PATH_4 = "emote_videos/emote4.mp4"  # Mouth open + hand near face
VIDEO_PATH_5 = "emote_videos/emote5.mp4"  # Booty shake
VIDEO_PATH_6 = "emote_videos/emote6.mp4"  # Hands descend to hips

TRIGGER_COOLDOWN = 10
SUSTAIN_FRAMES = 7

# Thresholds for gesture detection
HAND_NEAR_EYE_DISTANCE = 0.2  # Max distance for hand near eyes (Emote 1)
FINGER_NEAR_JAWLINE_DISTANCE = 0.08  # Max distance for finger near jawline (Emote 2)
HAND_VERTICAL_MOVEMENT_THRESHOLD = 0.10  # Vertical movement threshold (Emote 3)
HAND_NEAR_FACE_DISTANCE = 0.22  # Max distance from index fingertip to nose (Emote 4)
MOUTH_OPEN_THRESHOLD = 0.02  # Mouth opening threshold (Emote 4)
HAND_TRACKING_WINDOW = 10  # Frames to track for hand movement (Emote 3)
HAND_SWAP_POSITION_THRESHOLD = 0.5  # X position threshold for hand swap detection (Emote 5)
HAND_TO_HIP_DISTANCE_THRESHOLD = 0.12  # Max normalized distance from wrist to hip center (Emote 6)

# === Setup MediaPipe ===
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Detect up to 2 hands
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
cap = cv2.VideoCapture(0)

gesture_frames = 0
cooldown = 0

# Hand position tracking for Emote 3 (vertical movement detection)
hand_positions = [deque(maxlen=HAND_TRACKING_WINDOW) for _ in range(2)]  # Track up to 2 hands

# === Queues and flags for video frames ===
video_queues = [queue.Queue(maxsize=1) for _ in range(6)]
play_video_flags = [threading.Event() for _ in range(6)]

# === Function to read video frames in a separate thread ===
def video_reader(video_path, video_queue, play_flag):
    while True:
        play_flag.wait()  # wait until flagged
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            print("❌ Could not open video:", video_path)
            play_flag.clear()
            continue

        # Get video frame rate (FPS)
        fps = vid.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # Default to 30 FPS if unable to determine
        frame_delay = 1.0 / fps  # Calculate delay between frames for proper playback speed
        
        while vid.isOpened() and play_flag.is_set():
            ret, frame = vid.read()
            if not ret:
                break
            if not video_queue.full():
                video_queue.put(frame)
            time.sleep(frame_delay)  # Play at correct frame rate

        vid.release()
        play_flag.clear()  # done playing

# Start video reader threads for all 5 videos
threading.Thread(target=video_reader, args=(VIDEO_PATH_1, video_queues[0], play_video_flags[0]), daemon=True).start()
threading.Thread(target=video_reader, args=(VIDEO_PATH_2, video_queues[1], play_video_flags[1]), daemon=True).start()
threading.Thread(target=video_reader, args=(VIDEO_PATH_3, video_queues[2], play_video_flags[2]), daemon=True).start()
threading.Thread(target=video_reader, args=(VIDEO_PATH_4, video_queues[3], play_video_flags[3]), daemon=True).start()
threading.Thread(target=video_reader, args=(VIDEO_PATH_5, video_queues[4], play_video_flags[4]), daemon=True).start()
threading.Thread(target=video_reader, args=(VIDEO_PATH_6, video_queues[5], play_video_flags[5]), daemon=True).start()

# === Helper Functions ===
def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2) ** 0.5

def mouth_openness(face_landmarks):
    """Calculate mouth openness (vertical distance between lips)"""
    upper_lip = face_landmarks[13].y
    lower_lip = face_landmarks[14].y
    return abs(lower_lip - upper_lip)

# (removed unused brow helper)

# === Emote Detection Functions ===
def detect_emote1_both_hands_near_eyes(face_landmarks, hands_result):
    """
    Emote 1: Both hands near eyes
    Detection: Both hands must have fingertips near eyes
    """
    if not hands_result.multi_hand_landmarks or len(hands_result.multi_hand_landmarks) < 2:
        return False
    
    # Get eye landmarks
    left_eye = face_landmarks[33]   # Left eye corner
    right_eye = face_landmarks[362]  # Right eye corner
    
    hands_near_eyes_count = 0
    
    # Check each hand
    for hand_landmarks in hands_result.multi_hand_landmarks:
        # Use index fingertip
        index_tip = hand_landmarks.landmark[8]
        
        # Check distance to left eye
        dist_left = calculate_distance(index_tip, left_eye)
        # Check distance to right eye
        dist_right = calculate_distance(index_tip, right_eye)
        
        # If hand is near either eye
        if dist_left < HAND_NEAR_EYE_DISTANCE or dist_right < HAND_NEAR_EYE_DISTANCE:
            hands_near_eyes_count += 1
    
    return hands_near_eyes_count >= 2

def detect_emote2_finger_near_jawline(face_landmarks, hands_result):
    """
    Emote 2: One finger reaches the jawline
    Detection: Any fingertip near the jawline area
    """
    if not hands_result.multi_hand_landmarks:
        return False
    
    # Get jawline landmarks (bottom of face)
    jawline_landmarks = [face_landmarks[234], face_landmarks[365], face_landmarks[397], 
                         face_landmarks[288], face_landmarks[361], face_landmarks[454],
                         face_landmarks[152]]
    
    # Check each hand
    for hand_landmarks in hands_result.multi_hand_landmarks:
        # Use index fingertip
        index_tip = hand_landmarks.landmark[8]
        
        # Check if finger is near any jawline point
        for jawline_point in jawline_landmarks:
            distance = calculate_distance(index_tip, jawline_point)
            if distance < FINGER_NEAR_JAWLINE_DISTANCE:
                return True
    
    return False

def detect_emote3_hands_up_down_movement(face_landmarks, hands_result):
    """
    Emote 3: Both hands move up and down more than threshold
    Detection: Track hand vertical positions and detect movement
    """
    if not hands_result.multi_hand_landmarks or len(hands_result.multi_hand_landmarks) < 2:
        return False
    
    # Store current hand positions for tracking
    current_hand_positions = []
    for hand_landmarks in hands_result.multi_hand_landmarks:
        index_tip = hand_landmarks.landmark[8]
        current_hand_positions.append(index_tip.y)
    
    # Store positions for the oldest hand deque that has space
    if len(current_hand_positions) >= 2:
        for i, y_pos in enumerate(current_hand_positions[:2]):
            hand_positions[i].append(y_pos)
    
    # Check if both hands have enough history
    both_hands_detected = all(len(positions) >= HAND_TRACKING_WINDOW for positions in hand_positions[:len(current_hand_positions)])
    
    if not both_hands_detected:
        return False
    
    # Calculate vertical movement for each hand
    movements = []
    for positions in hand_positions[:len(current_hand_positions)]:
        if len(positions) >= 2:
            max_y = max(positions)
            min_y = min(positions)
            vertical_movement = abs(max_y - min_y)
            movements.append(vertical_movement)
    
    # Check if both hands moved more than threshold
    return all(mov > HAND_VERTICAL_MOVEMENT_THRESHOLD for mov in movements) and len(movements) >= 2

def detect_emote4_mouth_open_hand_near_face(face_landmarks, hands_result):
    """
    Emote 4: Mouth opens + hand within threshold to face
    Detection: Check mouth opening and EXACTLY ONE hand near face (uses index fingertip to nose distance)
    Require only 1 hand to avoid confusion with emote 1
    """
    # Check mouth is open
    mouth_opening = mouth_openness(face_landmarks)
    mouth_open = mouth_opening > MOUTH_OPEN_THRESHOLD
    
    if not mouth_open:
        return False

    # Check if exactly ONE hand is near face (exclude if both hands are near)
    if not hands_result.multi_hand_landmarks:
        return False
    
    # Get nose tip
    nose_tip = face_landmarks[1]
    
    hands_near_face_count = 0
    
    # Count how many hands are near the face
    for hand_landmarks in hands_result.multi_hand_landmarks:
        # Use index fingertip for more accurate detection
        index_tip = hand_landmarks.landmark[8]
        distance = calculate_distance(index_tip, nose_tip)
        
        if distance < HAND_NEAR_FACE_DISTANCE:
            hands_near_face_count += 1
    
    # Only trigger if EXACTLY ONE hand is near face
    return hands_near_face_count == 1

def detect_emote5_booty_shake(pose_result):
    """
    Emote 5: Booty shake (turn around by hip reversal)
    Detection: Use hips (23 left, 24 right). In a mirrored front camera,
    facing the camera typically has left_hip.x < right_hip.x. When turned
    around, the apparent ordering reverses (left_hip.x > right_hip.x).
    """
    if not pose_result or not pose_result.pose_landmarks:
        return False

    lm = pose_result.pose_landmarks.landmark
    left_hip = lm[23]
    right_hip = lm[24]

    # Require hips visible
    if left_hip.visibility < 0.5 or right_hip.visibility < 0.5:
        return False

    # In a mirrored front camera, your forward-facing pose may appear swapped.
    # Trigger only when turned around: left hip appears LEFT of right hip after turn.
    hips_reversed = left_hip.x < right_hip.x
    return hips_reversed

def detect_emote6_hands_to_hips(pose_result):
    """
    Emote 6: Hands to hips while facing forward
    Detection (simplified for 7-frame sustain):
      - Facing forward (opposite of Emote 5 hip-reversal)
      - Both wrists are within a distance threshold of the hip center
    """
    if not pose_result or not pose_result.pose_landmarks:
        return False

    lm = pose_result.pose_landmarks.landmark

    # Hips and wrists
    left_hip = lm[23]
    right_hip = lm[24]
    left_wrist = lm[15]
    right_wrist = lm[16]

    # Require visibility
    if (left_hip.visibility < 0.5 or right_hip.visibility < 0.5 or
        left_wrist.visibility < 0.5 or right_wrist.visibility < 0.5):
        return False

    # Facing forward in mirrored view: left hip appears to the RIGHT of right hip
    facing_forward = left_hip.x > right_hip.x
    if not facing_forward:
        return False

    # Hip center
    hip_center = type('obj', (object,), {
        'x': (left_hip.x + right_hip.x) / 2.0,
        'y': (left_hip.y + right_hip.y) / 2.0,
        'z': (left_hip.z + right_hip.z) / 2.0,
    })()

    # Distances from wrists to hip center
    left_dist = calculate_distance(left_wrist, hip_center)
    right_dist = calculate_distance(right_wrist, hip_center)

    return (left_dist < HAND_TO_HIP_DISTANCE_THRESHOLD and
            right_dist < HAND_TO_HIP_DISTANCE_THRESHOLD)

# === Main loop ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process face
    face_result = face_mesh.process(frame_rgb)
    
    # Process hands
    hands_result = hands.process(frame_rgb)
    
    # Process pose
    pose_result = pose.process(frame_rgb)
    
    gesture_detected = False
    detected_emote = 0  # 0 = none, 1-5 = emote number

    if face_result.multi_face_landmarks and hands_result:
        face_landmarks = face_result.multi_face_landmarks[0].landmark

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

        # Detect which emote is active (face/hand based)
        if detect_emote1_both_hands_near_eyes(face_landmarks, hands_result):
            detected_emote = 1
            gesture_detected = True
        elif detect_emote2_finger_near_jawline(face_landmarks, hands_result):
            detected_emote = 2
            gesture_detected = True
        elif detect_emote3_hands_up_down_movement(face_landmarks, hands_result):
            detected_emote = 3
            gesture_detected = True
        elif detect_emote4_mouth_open_hand_near_face(face_landmarks, hands_result):
            detected_emote = 4
            gesture_detected = True
    
    # Detect emote 5 (pose based)
    if not gesture_detected and detect_emote5_booty_shake(pose_result):
        detected_emote = 5
        gesture_detected = True
    # Detect emote 6 (hands descend to hips)
    if not gesture_detected and detect_emote6_hands_to_hips(pose_result):
        detected_emote = 6
        gesture_detected = True

    # Draw hand landmarks
    if hands_result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(hands_result.multi_hand_landmarks):
            # Get hand type (left or right)
            hand_type = hands_result.multi_handedness[idx].classification[0].label
            
            # Draw all hand landmarks
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)
            
            # Draw connections between landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2)
            )
            
            # Label hand type
            if hand_landmarks.landmark:
                label_x = int(hand_landmarks.landmark[0].x * frame.shape[1])
                label_y = int(hand_landmarks.landmark[0].y * frame.shape[0]) - 10
                cv2.putText(frame, hand_type, (label_x, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Draw pose landmarks (for emote 5/6 detection)
    if pose_result and pose_result.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            pose_result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
            mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 255), thickness=2)
        )
        
        # Highlight hips (emote 5) and wrists (emote 6)
        landmarks = pose_result.pose_landmarks.landmark

        # Hips
        lhx = int(landmarks[23].x * frame.shape[1]); lhy = int(landmarks[23].y * frame.shape[0])
        rhx = int(landmarks[24].x * frame.shape[1]); rhy = int(landmarks[24].y * frame.shape[0])
        cv2.circle(frame, (lhx, lhy), 10, (0, 0, 255), 2)
        cv2.circle(frame, (rhx, rhy), 10, (0, 0, 255), 2)

        # Wrists
        lwx = int(landmarks[15].x * frame.shape[1]); lwy = int(landmarks[15].y * frame.shape[0])
        rwx = int(landmarks[16].x * frame.shape[1]); rwy = int(landmarks[16].y * frame.shape[0])
        cv2.circle(frame, (lwx, lwy), 8, (255, 0, 0), 2)
        cv2.circle(frame, (rwx, rwy), 8, (0, 255, 0), 2)

    # Count consecutive frames
    if gesture_detected:
        gesture_frames += 1
    else:
        gesture_frames = 0

    # Trigger video if sustained
    if gesture_frames >= SUSTAIN_FRAMES and cooldown == 0 and detected_emote > 0:
        emote_names = ["", "Both Hands Near Eyes", "Finger Reaches Jawline", "Hands Up/Down Movement", "Mouth Open + Hand Near Face", "Booty Shake (Hips)", "Hands to Hips"]
        print(f"🎭 Emote {detected_emote} ({emote_names[detected_emote]}) detected! Playing video...")
        
        # Trigger the correct video based on detected emote (1-6 maps to indices 0-5)
        play_video_flags[detected_emote - 1].set()
        cooldown = TRIGGER_COOLDOWN
        gesture_frames = 0

    # Cooldown counter
    if cooldown > 0:
        cooldown -= 1

    # Add minimal status text (compact for split view)
    status_y = 30
    
    # Show which emote is detected
    if detected_emote > 0:
        cv2.putText(frame, f"Emote {detected_emote} Active", (10, status_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        status_y += 25
        cv2.putText(frame, f"Progress: {gesture_frames}/{SUSTAIN_FRAMES}", (10, status_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "Waiting for emote...", (10, status_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
    
    if cooldown > 0:
        status_y += 25
        cv2.putText(frame, f"Cooldown: {cooldown}", (10, status_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    
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
    
    # Create side-by-side display
    emote_size = 640  # Right box stays square
    camera_width = 960  # Camera box is wider (better aspect ratio for webcam)
    camera_height = 640
    
    # Resize camera frame to wider rectangle
    camera_display = cv2.resize(frame, (camera_width, camera_height))
    
    # Check if any video is playing
    video_frame = None
    playing_emote = 0
    
    for i, video_queue in enumerate(video_queues):
        if not video_queue.empty():
            video_frame = video_queue.get()
            playing_emote = i + 1
            break
    
    # Create right side (emote display or placeholder)
    if video_frame is not None:
        # Resize video to square
        emote_display = cv2.resize(video_frame, (emote_size, emote_size))
        # Add emote label overlay
        cv2.putText(emote_display, f"Emote {playing_emote}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    else:
        # Create placeholder - gradient background
        emote_display = np.zeros((emote_size, emote_size, 3), dtype=np.uint8)
        # Add gradient effect
        for y in range(emote_size):
            intensity = int(20 + (y / emote_size) * 30)
            emote_display[y, :] = (intensity, intensity, intensity)
        
        # Show emote status
        cv2.putText(emote_display, "EMOTE", (emote_size//2 - 120, emote_size//2 - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        if detected_emote > 0:
            emote_names = ["", "Both Hands Near Eyes", "Finger Reaches Jawline", 
                          "Hands Up/Down Movement", "Mouth Open + Hand Near Face", "Booty Shake (Hips)", "Hands to Hips"]
            cv2.putText(emote_display, f"{detected_emote} DETECTED", 
                       (emote_size//2 - 180, emote_size//2 + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(emote_display, f"Progress: {gesture_frames}/{SUSTAIN_FRAMES}", 
                       (emote_size//2 - 150, emote_size//2 + 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        else:
            cv2.putText(emote_display, "Waiting...", (emote_size//2 - 100, emote_size//2 + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 2)
    
    # Add divider line between camera and emote
    camera_with_divider = camera_display.copy()
    cv2.line(camera_with_divider, (camera_width-5, 0), (camera_width-5, camera_height), 
             (100, 100, 100), 5)
    
    # Combine camera and emote side by side
    combined_display = np.hstack([camera_with_divider, emote_display])
    
    # Show combined display
    cv2.imshow("Emote Detection System", combined_display)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()