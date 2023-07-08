import cv2
import os
import numpy as np
points = []

p1 = [(564, 645), (613, 686)]
p2 = [(773, 647), (807, 684)]
p3 = [(950, 646), (995, 683)]
p4 = [(1248, 650), (1284, 679)]

resolution = (1360, 2418)

#ratio_p = [[(0.4147058823529412, 0.2626137303556658), (0.45073529411764707, 0.2853598014888337)], [(0.5683823529411764, 0.2626137303556658), (0.5933823529411765, 0.2853598014888337)], [(0.6985294117647058, 0.2626137303556658), (0.7316176470588235, 0.2853598014888337)], [(0.7911764705882353, 0.2626137303556658), (0.9397058823529412, 0.2853598014888337)], [(1.2183823529411764, 0.2626137303556658), (1.2426470588235294, 0.2853598014888337)], [(1.3036764705882353, 0.2626137303556658), (1.326470588235294, 0.2853598014888337)]]

#for i in p:
 #   ratio_p.append([(i[0][0]/resolution[0], i[0][1]/resolution[1]), (i[1][0]/resolution[0], i[1][1]/resolution[1])])

#print(ratio_p)



def draw_rectangle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        # เพิ่มจุดที่ถูกคลิกลงในลิสต์
        points.append((x, y))

        # วาดสี่เหลี่ยมถ้ามีจุด 2 จุด
        if len(points) == 2:
            print(points)
            cv2.rectangle(resized_img, points[0], points[1], (255, 0, 0), 2)
            cv2.imshow('image', resized_img)

directory = 'images'
video_directory = 'video'
image_date = '2023-8-7'
#image_file = 'IMG_E0823.JPG'
image_list = ['IMG_E0823.JPG','IMG_E0824.JPG','IMG_E0825.JPG','IMG_E0826.JPG']
video_file = 'IMG_0827.MOV'
resized_video = 0.6296875
resized_imgs = 0.3

def image_detect_find_circles():
    p = [[(564, 635), (613, 690)], [(773, 635), (807, 690)], [(950, 635), (995, 690)], [(1076, 635), (1278, 690)],[(1657, 635), (1690, 690)],[(1773, 635), (1804, 690)]]
    for image_file in image_list:
        
        image_path = os.path.join(directory, image_date, image_file)
        
        # อ่านภาพ
        img = cv2.imread(image_path)
        
        resized_img = cv2.resize(img, (int(img.shape[1] * resized_imgs), int(img.shape[0] * resized_imgs)))
        print(resized_img.shape)
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)



        for i in p:
            color_img = resized_img[i[0][1]:i[1][1], i[0][0]:i[1][0]]
            image_crop = gray_img[i[0][1]:i[1][1], i[0][0]:i[1][0]]
            blur_img = cv2.GaussianBlur(image_crop, (5, 5), 0)
            ret, thresh = cv2.threshold(blur_img, 10, 255, cv2.THRESH_BINARY)
            circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=10, minRadius=0, maxRadius=0)
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for j in circles[0, :]:
                    print(j)
                    # วาดวงกลม
                    cv2.circle(color_img, (j[0], j[1]), j[2], (0, 255, 0), 1)
                    # วาดศูนย์กลางของวงกลม
                    cv2.circle(color_img, (j[0], j[1]), 1, (0, 0, 255), 1)
                #thresh = cv2.adaptiveThreshold(image_crop, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
                #ret, thresh = cv2.threshold(blur_img, 30, 255, cv2.THRESH_BINARY)
            #resized_img[i[0][1]:i[1][1], i[0][0]:i[1][0]] = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                    resized_img[i[0][1]:i[1][1], i[0][0]:i[1][0]] = color_img
                #cv2.rectangle(resized_img, i[0], i[1], (255, 0, 0), 2)

        # ผูกเมาส์กับภาพ
        cv2.namedWindow('image')
        #cv2.setMouseCallback('image', draw_rectangle)

        # แสดงภาพ
        cv2.imshow('image', resized_img)
        cv2.waitKey(0)

        # ปิดหน้าต่างภาพทั้งหมด
        cv2.destroyAllWindows()

def video_detect_contour():
    
    video_path = os.path.join(video_directory, video_file)
    print(video_path)
    p = [[(458, 640),(508, 700) ],[(705, 640),(738, 700) ],[(909, 640),(950, 700)],[(1051, 640),(1295, 700)],[(1727, 640),(1764, 700)],[(1868, 640),(1897, 700)]]
    cap = cv2.VideoCapture(video_path)
    # อัตราเฟรมต้นฉบับ (คุณต้องตั้งค่าตัวเลขนี้เองตามวิดีโอ)
    original_fps = 60

    # อัตราเฟรมที่คุณต้องการ
    desired_fps = 20

    # คำนวณว่าควรจะข้ามเฟรมกี่เฟรม
    skip_frames = round(original_fps / desired_fps)

    frame_count = 0
    frame_width = int(cap.get(3) * resized_video)
    frame_height = int(cap.get(4) * resized_video)
    print(frame_width, frame_height)
    out = cv2.VideoWriter('video_detect_contour.mp4', cv2.VideoWriter_fourcc('m','p','4','v'), 30, (frame_height ,frame_width))

    while cap.isOpened():
        ret, frame = cap.read()
        # ถ้าการอ่านเฟรมสำเร็จ ret จะเป็น True
        if not ret:
            break
        
        #if frame_count % skip_frames == 0:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        resized_img = cv2.resize(frame, (int(frame.shape[1] * resized_video), int(frame.shape[0] * resized_video)))
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.medianBlur(gray_img, ksize=15)
        for i in p:
            
            color_img = resized_img[i[0][1]:i[1][1], i[0][0]:i[1][0]]
            image_crop = gray_img[i[0][1]:i[1][1], i[0][0]:i[1][0]]
            #blur_img = cv2.GaussianBlur(image_crop, (5,5), 1)
            ret, thresh = cv2.threshold(image_crop, 10, 255, cv2.THRESH_BINARY)
            #thresh = cv2.bitwise_not(thresh)
            #kernel = np.ones((5,5),np.uint8)
            #thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            #thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            #resized_img[i[0][1]:i[1][1], i[0][0]:i[1][0]] = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # วนลูปเพื่อตรวจสอบทุก contours
            for contour in contours:
                # คำนวณเส้นรอบวง (perimeter)
                #perimeter = cv2.arcLength(contour, True)
                #hull = cv2.convexHull(contour)
                # หาวงกลมที่มีขนาดเล็กที่สุดที่สามารถครอบคลุม contour ได้
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)

                cv2.circle(color_img, center, radius, (0, 255, 0), 2)

                # ถ้าพื้นที่ของวงกลมประมาณ 90% ของพื้นที่ contour แสดงว่า contour เป็นวงกลม
                #try:
                #    if abs(1 - (cv2.contourArea(contour) / (np.pi * radius ** 2))) <= 0.3:
                #        cv2.circle(color_img, center, radius, (0, 255, 0), 2)
                #except:
                #    pass
            resized_img[i[0][1]:i[1][1], i[0][0]:i[1][0]] = color_img
            cv2.rectangle(resized_img, i[0], i[1], (255, 0, 0), 2)
        out.write(resized_img)
        cv2.imshow('Frame', resized_img)

        # หยุดเมื่อกด 'q'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        #frame_count += 1
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def video_detect_find_circles():
    
    video_path = os.path.join(video_directory, video_file)
    print(video_path)
    p = [[(458, 640),(508, 700) ],[(705, 640),(738, 700) ],[(909, 640),(950, 700)],[(1051, 640),(1295, 700)],[(1727, 640),(1764, 700)],[(1868, 640),(1897, 700)]]
    cap = cv2.VideoCapture(video_path)
    # อัตราเฟรมต้นฉบับ (คุณต้องตั้งค่าตัวเลขนี้เองตามวิดีโอ)
    original_fps = 60

    # อัตราเฟรมที่คุณต้องการ
    desired_fps = 20

    # คำนวณว่าควรจะข้ามเฟรมกี่เฟรม
    skip_frames = round(original_fps / desired_fps)

    frame_count = 0
    frame_width = int(cap.get(3) * resized_video)
    frame_height = int(cap.get(4) * resized_video)
    print(frame_width, frame_height)
    out = cv2.VideoWriter('video_detect_find_circles.mp4', cv2.VideoWriter_fourcc('m','p','4','v'), 30, (frame_height ,frame_width))

    while cap.isOpened():
        ret, frame = cap.read()
        # ถ้าการอ่านเฟรมสำเร็จ ret จะเป็น True
        if not ret:
            break
        
        #if frame_count % skip_frames == 0:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        resized_img = cv2.resize(frame, (int(frame.shape[1] * resized_video), int(frame.shape[0] * resized_video)))
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        for i in p:
            
            color_img = resized_img[i[0][1]:i[1][1], i[0][0]:i[1][0]]
            image_crop = gray_img[i[0][1]:i[1][1], i[0][0]:i[1][0]]
            blur_img = cv2.GaussianBlur(image_crop, (5, 5), 0)
            ret, thresh = cv2.threshold(blur_img, 10, 255, cv2.THRESH_BINARY)
            circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=10, minRadius=0, maxRadius=0)
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for j in circles[0, :]:
                    cv2.circle(color_img, (j[0], j[1]), j[2], (0, 255, 0), 2)
                    cv2.circle(color_img, (j[0], j[1]), 1, (0, 0, 255), 1)
                    resized_img[i[0][1]:i[1][1], i[0][0]:i[1][0]] = color_img
            cv2.rectangle(resized_img, i[0], i[1], (255, 0, 0), 2)
        out.write(resized_img)
        cv2.imshow('Frame', resized_img)

        # หยุดเมื่อกด 'q'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        #frame_count += 1
    cap.release()
    out.release()
    cv2.destroyAllWindows()

video_detect_contour()