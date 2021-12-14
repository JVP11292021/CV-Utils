# **get_contour**

Here you can see a simple example of the get_contour function. This function gets the contours within an image and returns them.

`

    from Models.models import get_contours, img_stacker
    from Models.constants import ALL_CONTOURS
    import cv2 as cv
    import numpy as np
    
    img = cv.imread("getty_517194189_373099.jpg")
    img_contour = img.copy()
    
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (7, 7), 1)
    img_canny = cv.Canny(img_blur, 50, 50)
    img_blank = np.zeros_like(img)
    
    _, all_contours = get_contours(img_canny, img_contour, ALL_CONTOURS, thickness=5)
    
    img_stack = img_stacker(.2, ([img_gray, img, img_blur], [img_canny, img_contour, all_contours]))
    
    cv.imshow("blur", img_stack)
    
    
    cv.waitKey(0)

`
