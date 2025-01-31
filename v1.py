import cv2
import face_recognition
import numpy as np
import logging
import os

# Setup logging for better monitoring
logging.basicConfig(filename='face_recognition.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

# Check for webcam availability
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    logging.error("Webcam not found!")
    raise Exception("Webcam not found!")

# Load known faces from a persistent database or pre-trained model (using a database or directory)
def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    try:
        obama_image = face_recognition.load_image_file("obama.jpg")
        obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
        known_face_encodings.append(obama_face_encoding)
        known_face_names.append("Barack Obama")

        biden_image = face_recognition.load_image_file("biden.jpg")
        biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
        known_face_encodings.append(biden_face_encoding)
        known_face_names.append("Joe Biden")
        
        logging.info("Known faces loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading faces: {str(e)}")
        raise e
    return known_face_encodings, known_face_names

known_face_encodings, known_face_names = load_known_faces()

# Process video frames with optimizations
process_this_frame = True
while True:
    try:
        ret, frame = video_capture.read()
        if not ret:
            logging.error("Failed to capture video frame")
            break

        if process_this_frame:
            # Resize for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Find faces
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                face_names.append(name)

        process_this_frame = not process_this_frame

        # Display results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    except Exception as e:
        logging.error(f"Error during face recognition process: {str(e)}")
        break

# Release resources on exit
video_capture.release()
cv2.destroyAllWindows()
logging.info("Application terminated gracefully.")
