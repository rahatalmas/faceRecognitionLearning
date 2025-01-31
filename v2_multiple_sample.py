import face_recognition
import cv2
import numpy as np
import os

# Folder to store images
known_face_encodings = []
known_face_names = []

# Load multiple images for the same person
def load_face_images(image_paths, person_name):
    encodings = []
    for image_path in image_paths:
        # Load the image file
        image = face_recognition.load_image_file(image_path)
        
        # Get the face encoding of the image
        face_encoding = face_recognition.face_encodings(image)
        
        # If a face is found, append the encoding
        if face_encoding:
            encodings.append(face_encoding[0])
        else:
            print(f"No face found in image: {image_path}")
    
    # Store the encodings and name
    for encoding in encodings:
        known_face_encodings.append(encoding)
        known_face_names.append(person_name)

# Example: Load images of Barack Obama
image_paths = ["obama1.jpg", "obama2.jpg", "obama3.jpg"]
load_face_images(image_paths, "Barack Obama")

# Example: Load images of Joe Biden
image_paths = ["biden1.jpg", "biden2.jpg"]
load_face_images(image_paths, "Joe Biden")

# Initialize the video capture (webcam)
video_capture = cv2.VideoCapture(0)

process_this_frame = True
while True:
    ret, frame = video_capture.read()
    
    if not ret:
        break

    if process_this_frame:
        # Resize the frame to speed up processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Check for matches with any known face encodings
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match is found, use the name of the matched person
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale the face location back up since the frame was resized
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with the person's name
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow("Video", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources and close windows
video_capture.release()
cv2.destroyAllWindows()
