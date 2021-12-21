# **CV-Utils**

This project was created for newer opencv-python developers. In this library there are a number of utility functions for opencv-python, there are also a couple of advanced models in here, like Handtracking, facedetection and facerecognition.

## **Examples of alpha version**

**Face recognition**
`
    
    import Models as mod
    from Models.models import face_locations, face_encodings, key_pressed
    import cv2 as cv
    import traceback

    capture = cv.VideoCapture(0)
    # Initializing the fps
    fps = mod.FPS()

    recognizer = mod.FaceRecognition("model_data")

    # finding the face encodings(features in the face like eyes mouths and measurements)
    try:
        encode_known_faces = recognizer.find_encodings(images=recognizer.saved_images)
    except Exception:
        print(traceback.format_exc())
    # the webcam frame loop

    while True:
        isTrue, frame = capture.read()
    
        # resized frame
        res_img = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # resized image with the converted color channels
        r_res_img = cv.cvtColor(res_img, cv.COLOR_BGR2RGB)
         
        # returning a tuple of all the face locations in the image
        faces_curr_frame = face_locations(r_res_img)
        encodes_curr_frame = face_encodings(r_res_img, faces_curr_frame)
        
        recognizer.find_faces(frame=frame,
                              face_encodings=encodes_curr_frame,
                              face_locations=faces_curr_frame,
                              encode_known_faces=encode_known_faces,
                              draw=True)
        
        # Drawing the fps on the screen
        fps.draw_fps(frame, (70, 40), cv.FONT_HERSHEY_PLAIN)
        
        cv.imshow("Face recognition", frame)
        cv.waitKey(1)
        
        # Using this function to close the loop. It returns cv.waitKey(1) & 0xFF == ord(key)
        if key_pressed("q"):
            break

    capture.release()
    cv.destroyAllWindows()
`

**Face detection**
`
    
    import Models as mod
    from Models.models import face_locations, face_encodings, key_pressed
    import cv2 as cv
    import traceback

    capture = cv.VideoCapture(0)
    # Initializing the fps
    fps = mod.FPS()

    detector = mod.FaceDetection()

    # the webcam frame loop
    while True:
        isTrue, frame = capture.read()

        # frame with the converted color channels
        recoloured_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # returning a tuple of all the face locations in the image
        faces_curr_frame = face_locations(recoloured_frame)
        encodes_curr_frame = face_encodings(recoloured_frame, faces_curr_frame)

        detector.detect_faces(frame=frame,
                              face_locations=faces_curr_frame,
                              face_encodings=encodes_curr_frame,
                              draw=True)

        # Drawing the fps on the screen
        fps.draw_fps(frame, (70, 40), cv.FONT_HERSHEY_PLAIN)

        cv.imshow("Face detection", frame)
        
        # Using this function to close the loop. It returns cv.waitKey(1) & 0xFF == ord(key)
        if key_pressed("q"):
            break

    capture.release()
    cv.destroyAllWindows()
`

**Hand tracking**
`
    
    import Models as mod
    from Models.models import key_pressed
    import cv2 as cv

    capture = cv.VideoCapture(0)
    # Initializing the fps
    fps = mod.FPS()

    tracker = mod.HandTracker()

    while True:
        isTrue, frame = capture.read()
        
        # Finding the hands present in the frame and if draw paramater is true using the standard drawing
        frame = tracker.find_hands(frame)
        # Finding all the landmarks in the image if draw parameter is true then show all landmark points by number
        landmarks = tracker.find_position(frame, draw=False)
        
        # If you want to work with specific landmarks gotta check if the length does not equal 0 first
        if len(landmarks) != 0:
            if landmarks[12] is not None:
                print(print(landmarks[12]))
        
        # Drawing the fps on the screen
        fps.draw_fps(frame, (70, 40), cv.FONT_HERSHEY_PLAIN)


        cv.imshow("Handtracker", frame)

        # Using this function to close the loop. It returns cv.waitKey(1) & 0xFF == ord(key)
        if key_pressed('q'):
            break

    capture.release()
    cv.destroyAllWindows()
`

**Object detection**

`

    import cv2
    import Models as mod
    from Models import key_pressed
    
    
    modelFile = "resources/MobileNet_detection_2018/frozen_inference_graph.pb"
    configFile = "resources/MobileNet_detection_2018/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
    classFile = "resources/MobileNet_detection_2018/coco_class_labels.txt"
    
    detector = mod.MobNetObjectDetection(classFile, configFile, modelFile)
    path = "resources/MobileNet_detection_2018/test_data/videoplayback.mp4"
    cap = cv2.VideoCapture(path)
    
    while True:
        success, frame = cap.read()
    
        if success:
            detected_objects = detector.pre_process(frame, scale_factor=0.78)
    
            detector.display_cv_objects(frame, detected_objects)
    
            if key_pressed('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
`