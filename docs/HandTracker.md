# **Hand tracker**

In this file i'll explain how you can use the hand tracker module in your own computer vision projects.

First we need to make some imports. We import the custom package with Models.models. We also import some functions from Models.models. Third we import opencv the computervision library.

`

    import Models.models as mod
    from Models import close_win
    import cv2 as cv

`

Next we instantiate a capture, fps and a HandTracking object. The constructor needs no parameters to work properly.


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

        if close_win('q'):
            break

    capture.release()
    cv.destroyAllWindows()

`

Next we use the tracker.find_hands() function to find the hands in the current frame. This function finds all the hands present in the frame, it takes in the frame as parameter and returns a new frame. There ia also an optional parameter named draw. If this parameter is set to true then it uses the standard drawing. The tracker.find_position() function finds all the landmarks on the hand. This function returns all the image landmarks in list format, it takes in two parameters a frame and a draw flag. If the draw flag = true then it draws all the landmarks by number on the hand.

`

    while True:
        isTrue, frame = capture.read()

        frame = tracker.find_hands(frame)
        landmarks = tracker.find_position(frame, draw=False)

        cv.imshow("Test", frame)
    
        if close_win('q'):
            break

    capture.release()
    cv.destroyAllWindows()

`

Next we can use the landmarks variable to do some checks. With this landmark variable you can specify which landmark you want to use.

`

    while True:
        isTrue, frame = capture.read()

        frame = tracker.find_hands(frame)
        landmarks = tracker.find_position(frame, draw=False)

        if len(landmarks) != 0:
            if landmarks[12] is not None:
                print(print(landmarks[12]))

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
    from Models.models import face_locations, face_encodings, close_win
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
        if close_win('q'):
            break

    capture.release()
    cv.destroyAllWindows()

`