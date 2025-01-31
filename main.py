# # from PIL import Image
# # import face_recognition

# # # Load the jpg file into a numpy array
# # image = face_recognition.load_image_file("almas3.jpg")

# # # Find all the faces in the image using the default HOG-based model.
# # # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
# # # See also: find_faces_in_picture_cnn.py
# # face_locations = face_recognition.face_locations(image)

# # print("I found {} face(s) in this photograph.".format(len(face_locations)))

# # for face_location in face_locations:

# #     # Print the location of each face in this image
# #     top, right, bottom, left = face_location
# #     print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

# #     # You can access the actual face itself like this:
# #     face_image = image[top:bottom, left:right]
# #     pil_image = Image.fromarray(face_image)
# #     pil_image.show()

# import cv2 as cv
# import face_recognition

# # Load and process the first image
# img = cv.imread("aslam.jpg")
# img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# img_encoding = face_recognition.face_encodings(img_rgb)[0]

# # Load and process the second image
# #img2 = cv.imread("almas3.jpg")
# img2 = cv.imread("aslam2.jpg")
# img_rgb2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
# img_encoding2 = face_recognition.face_encodings(img_rgb2)[0]

# # Compare the faces
# result = face_recognition.compare_faces([img_encoding], img_encoding2)
# print(result)

# # Draw bounding boxes around faces in the first image
# face_locations = face_recognition.face_locations(img_rgb)

# for face_location in face_locations:
#     top, right, bottom, left = face_location
#     cv.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
#     #face = img[top:bottom,left:right]
#     #cv.imshow("face",face)
# # Draw bounding boxes around faces in the second image
# face_locations2 = face_recognition.face_locations(img_rgb2)
# for face_location in face_locations2:
#     top, right, bottom, left = face_location
#     cv.rectangle(img2, (left, top), (right, bottom), (0, 255, 0), 2)

# # Display the images with bounding boxes
# # width = 800
# # height = int(img.shape[0] * (width / img.shape[1]))  # Keep the aspect ratio
# # resized_img = cv.resize(img, (width, height))
# cv.imshow("display1", img)
# cv.imshow("display2", img2)

# # Wait for a key press and close windows
# cv.waitKey(0)
# cv.destroyAllWindows()


import face_recognition
import cv2
import numpy as np

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()