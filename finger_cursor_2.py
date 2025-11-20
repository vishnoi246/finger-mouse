import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Get screen size for coordinate mapping
screen_w, screen_h = pyautogui.size()

def count_fingers(hand_landmarks, handedness):
    fingers = []
    is_right_hand = handedness == "Right"

    # Thumb
    if is_right_hand:
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:
        if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
            fingers.append(1)
        else:
            fingers.append(0)

    # Other four fingers
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    for tip, pip in zip(tips, pips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)
    
    return fingers

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,  
    min_tracking_confidence=0.5,   
    model_complexity=0 
)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)            


pyautogui.FAILSAFE = False

while cap.isOpened():
    ref, frame = cap.read()
    
    if not ref:
        continue
    
    h, w, c = frame.shape
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = hands.process(rgb_frame)
    
    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, hand_handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            handedness = hand_handedness.classification[0].label
            fingers = count_fingers(hand_landmarks, handedness)
            
           
            if fingers and sum(fingers) == 1:
                idx = fingers.index(1)
                finger_tips = [4, 8, 12, 16, 20]
                tip_landmark = hand_landmarks.landmark[finger_tips[idx]]

                screen_x = int(tip_landmark.x * screen_w)
                screen_y = int(tip_landmark.y * screen_h)

                pyautogui.moveTo(screen_x, screen_y)
                
                # Draw tracking circle
                x = int(tip_landmark.x * w)
                y = int(tip_landmark.y * h)
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
    
    
    cv2.putText(frame, f'Press Q to quit', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Hand Tracking Cursor Control', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()