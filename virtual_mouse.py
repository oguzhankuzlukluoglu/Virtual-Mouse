import cv2
import mediapipe as mp
import pyautogui
import math

# Mouse sensitivity settings (adjust as needed)
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
SMOOTHING = 2  # Higher = smoother but slower movement

# MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)  # Track only one hand
mp_draw = mp.solutions.drawing_utils

# Initialize Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Variables for mouse movement
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)  # Mirror effect
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks (optional)
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get finger landmarks
            landmarks = hand_landmarks.landmark
            index_tip = landmarks[8]  # Index finger tip
            thumb_tip = landmarks[4]  # Thumb tip

            # Map hand position to screen coordinates
            curr_x = int(index_tip.x * SCREEN_WIDTH)
            curr_y = int(index_tip.y * SCREEN_HEIGHT)

            # Smooth mouse movement
            move_x = prev_x + (curr_x - prev_x) / SMOOTHING
            move_y = prev_y + (curr_y - prev_y) / SMOOTHING

            # Move the mouse
            pyautogui.moveTo(move_x, move_y)
            prev_x, prev_y = move_x, move_y

            # Click detection (thumb and index finger close)
            distance = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
            if distance < 0.03:  # Threshold (adjustable)
                pyautogui.click()
                cv2.putText(img, "CLICKED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the camera feed
    cv2.imshow("Virtual Mouse - Press 'Q' to Quit", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()