import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

dot_specs = mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=4)
connection_specs = mp_drawing.DrawingSpec(color=(255,0,0), thickness=4, circle_radius=2)

cap = cv2.VideoCapture(3) #webcam 3 start 0

cap.set(3, 1920)
cap.set(4, 1280)


#initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        #make detections
        results = holistic.process(image)
        #print(results.pose_landmarks)
        
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        #draw face detections
        #mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
        
        #draw pose detections
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, dot_specs, connection_specs)
        
        #draw right hand detection landmarks
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, dot_specs, connection_specs)
        
        #draw left hand detection landmarks
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, dot_specs, connection_specs)
        
        
        
        cv2.imshow('Holistic Model Detections', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()