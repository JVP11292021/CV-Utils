# **Face detection**

In this file i'll explain how you can use the face detection module in your own computer vision projects.

First we need to make some imports. We import the custom package with Models.models. We also import some functions from Models.models. Third we import opencv the computervision library.

`

    import Models as mod
    from Models import key_pressed, face_locations, face_encodings
    import cv2 as cv

`

Next we instantiate a capture, fps and a FaceDetection object. The constructor needs no parameters to work properly.


`

    capture = cv.VideoCapture(0)
    fps = mod.FPS()

    tracker = mod.HandTracker()

`

Next we create the standard opencv frame loop. Firstly we save the frame and the is_frame_running flag variables. Then we show the frame with cv.imshow(title, frame). Then we can break out of the loop with the close_win(key) function. This function was created to save beginner developers the headache of writing some complex logic. And lastly we release the capture and destroy the window.

`
    
    while True:
        isTrue, frame = capture.read()
        
        cv.imshow("Test", frame)

        if key_pressed('q'):
            break

    capture.release()
    cv.destroyAllWindows()

`

Next we convert the color channels of the frame from BGR to RGB. We also get the locations of the faces in the current frame, and the encodings of the face in the current frame.

`
    
    while True:
        isTrue, frame = capture.read()
        
        recoloured_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        faces_curr_frame = face_locations(recoloured_frame)
        encodes_curr_frame = face_encodings(recoloured_frame, faces_curr_frame)

        cv.imshow("Test", frame)

        if key_pressed('q'):
            break

    capture.release()
    cv.destroyAllWindows()

`

Next we use the detector.detect_faces() function to find the faces within the current frame. The function needs: The frame, the current face locations, the current face encodings as parameters. There is also an optional parameter you can specify and that is the draw parameter. If set to true the program automatically uses the standard drawing to show the user which faces it detects. 

`
    
    while True:
        isTrue, frame = capture.read()
        
        recoloured_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        faces_curr_frame = face_locations(recoloured_frame)
        encodes_curr_frame = face_encodings(recoloured_frame, faces_curr_frame)
        
        detector.detect_faces(frame=frame,
                              face_locations=faces_curr_frame,
                              face_encodings=encodes_curr_frame,
                              draw=True)

        cv.imshow("Test", frame)

        if key_pressed('q'):
            break

    capture.release()
    cv.destroyAllWindows()

`

After you've added all these steps the model should be working. Now we can add something optional for the model and that's  the fps. We can use the .draw_fps() function to display the current fps. The first parameter it needs is the frame, the second is the position in tuple format, the last required parameter is the font we want to use.

`

    while True:
        isTrue, frame = capture.read()

        res_img = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)
        r_res_img = cv.cvtColor(res_img, cv.COLOR_BGR2RGB)

        faces_curr_frame = face_locations(r_res_img)
        encodes_curr_frame = face_encodings(r_res_img, faces_curr_frame)

        recognizer.find_faces(frame=frame,
                              face_encodings=encodes_curr_frame,
                              face_locations=faces_curr_frame,
                              encode_known_faces=encode_known_faces,
                              draw=True)

        fps.draw_fps(frame, (70, 40), cv.FONT_HERSHEY_PLAIN)

        cv.imshow("Test", frame)
    
        if key_pressed('q'):
            break

    capture.release()
    cv.destroyAllWindows()
`

### **Complete program**

Note make sure to add cv.imshow() last.

Note you can use help(object) to view the descriptions within the console


`
    
    import Models as mod
    from Models.models import face_locations, face_encodings, key_pressed
    import cv2 as cv

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