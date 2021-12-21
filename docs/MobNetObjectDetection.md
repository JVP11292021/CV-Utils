# **MobNetObjectDetection**

### **With opencv video capture**

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

### **With static images opencv**

`

    import Models as mod
    import cv2 as cv
    
    modelFile = "resources/MobileNet_detection_2018/frozen_inference_graph.pb"
    configFile = "resources/MobileNet_detection_2018/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
    classFile = "resources/MobileNet_detection_2018/coco_class_labels.txt"
    
    detector = mod.MobNetObjectDetection(model_file=modelFile, config_file=configFile, class_file=classFile)
    
    img = cv.imread("resources/MobileNet_detection_2018/test_data/street.jpg")
    
    detected_objects = detector.pre_process(img, scale_factor=0.7)
    detector.display_cv_objects(img, detected_objects, min_confidence=0.4)
    
    cv.waitKey(0)
`

### **With static images matplotlib**

`

    import Models as mod
    import cv2 as cv
    
    modelFile = "resources/MobileNet_detection_2018/frozen_inference_graph.pb"
    configFile = "resources/MobileNet_detection_2018/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
    classFile = "resources/MobileNet_detection_2018/coco_class_labels.txt"
    
    detector = mod.MobNetObjectDetection(model_file=modelFile, config_file=configFile, class_file=classFile)
    
    img = cv.imread("resources/MobileNet_detection_2018/test_data/street.jpg")
    
    detected_objects = detector.pre_process(img, scale_factor=0.7)
    detector.display_plt_objects(img, detected_objects, min_confidence=0.4)
    
    cv.waitKey(0)
`