# **Face recognition**

In this file i'll explain how you can use the face recognition module in your own computer vision projects.

First we need to make some imports. We import the custom package with Models.models. We also import some functions from Models.models. Third we import opencv the computervision library and lastly we import traceback for good error handling.

`

    import Models.models as mod
    from Models import face_locations, face_encodings, close_win
    import cv2 as cv
    import traceback

`

Next we instantiate a capture, fps and a FaceRecognition object. Make sure that you pass the path to the dataset into the FaceRecognition constructor. In our case that's in a folder named "model_data".

`

    capture = cv.VideoCapture(0)
    fps = mod.FPS()

    recognizer = mod.FaceRecognition("model_data")

`

Next we search for encodings within the face.  encoding are features in the face like eyes mouths and measurements. We access the find_encodings function from the recognizer object. We also embed the code within a try except clause. We do this in case the folder is empty then it won't exit the program.

`

    try:
        encode_known_faces = recognizer.find_encodings(images=recognizer.saved_images)
    except Exception:
        print(traceback.format_exc())

`

Next we create the standard opencv frame loop. Firstly we save the frame and the is_frame_running flag variables. Then we show the frame with cv.imshow(title, frame). Then we can break out of the loop with the close_win(key) function. This function was created to save beginner developers the headache of writing some complex logic. And lastly we release the capture and destroy the window.

`
    
    while True:
        isTrue, frame = capture.read()
        
        cv.imshow("Test", frame)

        if close_win('q'):
            break

    capture.release()
    cv.destroyAllWindows()

`

Next we need to resize the frame to 4x less. We do this with cv.resize(fx=0.25, fy=0.25). Secondly we convert the colors from BGR to the standard RGB format.

`

    while True:
        isTrue, frame = capture.read()

        res_img = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)
        r_res_img = cv.cvtColor(res_img, cv.COLOR_BGR2RGB)

        cv.imshow("Test", frame)
    
        if close_win('q'):
            break

    capture.release()
    cv.destroyAllWindows()

`

Next we add the current face encoding and the face locations functions. The face_location function returns a tuple of all the face locations in the image. The face_encoding needs a resized frame and the face_locations tuple.

`

    while True:
        isTrue, frame = capture.read()

        res_img = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)
        r_res_img = cv.cvtColor(res_img, cv.COLOR_BGR2RGB)

        faces_curr_frame = face_locations(r_res_img)
        encodes_curr_frame = face_encodings(r_res_img, faces_curr_frame)

        cv.imshow("Test", frame)
    
        if close_win('q'):
            break

    capture.release()
    cv.destroyAllWindows()

`

Next we add the find_faces function from the FaceRecognition class. This function needs a couple parameters to work properly. First it needs the frame, second it needs the encodings from the current frame, third it needs the face_locations, fourth it needs the encodings from the known faces, and lastly an optional parameter if you want to use the standard drawing or not. 

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

        cv.imshow("Test", frame)
    
        if close_win('q'):
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
    
        if close_win('q'):
            break

    capture.release()
    cv.destroyAllWindows()
`

### **Complete program**

Note make sure to add cv.imshow() last.

Note you can use help(object) to view the descriptions within the console

`

    import Models.models as mod
    from Models import face_locations, face_encodings, close_win
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
        if close_win("q"):
            break

    capture.release()
    cv.destroyAllWindows()

`