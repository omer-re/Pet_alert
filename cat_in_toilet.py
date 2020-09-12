#!/usr/bin/env python
# coding: utf-8

# # YOLO v3 Object Detection
# 
# Let's see how to use the state of the art in object detection! Please make sure to watch the video, there is no code along here, since we can't reasonably train the YOLOv3 network ourself, instead we will use a pre-established version.
# 
# CODE SOURCE: https://github.com/xiaochus/YOLOv3
# 
# REFERENCE (for original YOLOv3): 
# 
#         @article{YOLOv3,  
#               title={YOLOv3: An Incremental Improvement},  
#               author={J Redmon, A Farhadi },
#               year={2018} 
# --------
# ----------
# ## YOU MUST WATCH THE VIDEO LECTURE TO PROPERLY SET UP THE MODEL AND WEIGHTS. THIS NOTEBOOK WON'T WORK UNLESS YOU FOLLOW THE EXACT SET UP SHOWN IN THE VIDEO LECTURE.
# -------
# -------

# In[3]:


import os
import time
import cv2
import numpy as np
from model.yolo_model import YOLO


# In[4]:


def process_image(img):
    """Resize, reduce and expand image.

    # Argument:
        img: original image.

    # Returns
        image: ndarray(64, 64, 3), processed image.
    """
    image = cv2.resize(img, (416, 416),
                       interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)

    return image


# In[5]:


def get_classes(file):
    """Get classes name.

    # Argument:
        file: classes name for database.

    # Returns
        class_names: List, classes name.

    """
    with open(file) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    return class_names


# In[6]:


def draw(image, boxes, scores, classes, all_classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2 ,(0, 0, 255), 3,
                    cv2.LINE_AA)

       # print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        #print('box coordinate x,y,w,h: {0}'.format(box))

    print()


# In[7]:


# function to find if  
# given point lies inside  
# a given rectangle or not. 
def FindPoint2(left, bottom, right, top, x, y,w,h) : 
    px1, py1, px2, py2 = left, top, right, bottom
    hx1, hy1, hx2, hy2 = x,y,x+w,y+h
    if (hx1 >= px1 and hy1 >= py1) or (hx2 <= px2 and hy2 <=py2):
        print("True!")
        return True
    else:
        return False

def FindPoint(left, bottom, right, top, x, y,w,h) : 
    try:
        cat_x=int((2*x+w)/2)
        cat_y=int((2*y+h)/2)
        if cat_x>left and cat_y<top and cat_x<right and cat_y>bottom:
            print("True!")
            return True
        else:
            return False
    except:
        print("something wrong")
        return False

    
    
def only_if_max(curr_val, new_val):
   
    if curr_val==None or new_val > curr_val :

        return new_val
    return curr_val
    
def only_if_min(curr_val, new_val):
    if curr_val==None or new_val < curr_val:
        return new_val
    return curr_val


# In[8]:


# In[22]:



def draw2(image, boxes, scores, classes, all_classes, bad_guy, forbbiden_place):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    cat_exsists=0
    toilet_exists=0
    toilet_left=None
    toilet_right=None
    toilet_bottom=None
    toilet_top=None 
    tags=[]
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))
        tags.append(all_classes[cl])
        if all_classes[cl]==bad_guy:
            cv2.rectangle(image, (top, left), (right, bottom), (0, 0, 255), 2)
            cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1 ,(0, 0, 255), 1,
                    cv2.LINE_AA)
        
        if all_classes[cl]==forbbiden_place:
            toilet_exists=1
            toilet_exists=0
            toilet_left=only_if_min(toilet_left, left)
            toilet_right=only_if_max(toilet_right,right)
            toilet_bottom=only_if_max(toilet_bottom, bottom)
            toilet_top=only_if_min(toilet_top, top)
            text_hieght=  int((toilet_bottom+toilet_top)/2)
            cat_x=int((2*x+w)/2)
            cat_y=int((2*y+h)/2)
            #print("toilet")
        
        if toilet_left: #prevent nonetype
            cv2.rectangle(image, (toilet_top, toilet_left), (toilet_right, toilet_bottom), (200, 0, 0), 2)
        if forbidden_place and bad_guy in tags:
            if bad_guy in tags:
                if FindPoint(toilet_left,toilet_top,toilet_right,toilet_bottom, x,y,w,h):
                    #image = cv2.circle(image, (cat_y,cat_x) , 2, (0, 0, 100), 2 )
                    text = str(str(bad_guy).upper()+" IS ON THE "+str(forbbiden_place).upper())
                    print(text)
                    #print("CAT IN THE TOILET")
                    image = cv2.circle(image, (cat_x,cat_y) , 4, (0, 0, 255), 4 )
                    text_y=text_hieght
                    text_x=int((cat_x/2))
                    # font 
                    font = cv2.FONT_HERSHEY_SIMPLEX 
                    
                    # org 
                    org = (text_x-100, text_y ) 

                    # fontScale 
                    fontScale = 1

                    # Blue color in BGR 
                    color = (50, 0, 255) 

                    # Line thickness of 2 px 
                    thickness = 2

                    # Using cv2.putText() method 
                    
                    y0, dy = text_y, 150
                    #for i, line in enumerate(text.split('\n')):
                    #        y = y0 + i*dy
                    #        cv2.putText(image, line, (text_x, y ), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv2.LINE_AA)
                    image = cv2.putText(image,  text, org, font, fontScale, color, thickness, cv2.LINE_AA) 

                    #cv2.putText(image,'{0} {1:.2f}'.format( "CAT IN THE TOILET"),(bottom, left +6), cv2.FONT_HERSHEY_SIMPLEX, 10 ,(0, 0, 255), 8, cv2.LINE_AA)
                
        #print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        #print('box coordinate x,y,w,h: {0}'.format(box))
    



# In[17]:


def detect_image(image, yolo, all_classes, bad_guy, forbbiden_place):
    """Use yolo v3 to detect images.

    # Argument:
        image: original image.
        yolo: YOLO, yolo model.
        all_classes: all classes name.

    # Returns:
        image: processed image.
    """
    pimage = process_image(image)

    start = time.time()
    boxes, classes, scores = yolo.predict(pimage, image.shape)
    end = time.time()

    print('frame process time: {0:.2f}s'.format(end - start))
    
    if boxes is not None:
        draw2(image, boxes, scores, classes, all_classes, bad_guy, forbbiden_place)

    return image


# In[18]:


def detect_video(video, yolo, all_classes, bad_guy, forbbiden_place):
    """Use yolo v3 to detect video.

    # Argument:
        video: video file.
        yolo: YOLO, yolo model.
        all_classes: all classes name.
    """
    print(video)
    # FILE SOURCE
    #video_path = os.path.join("videos", "test", video)
    #camera = cv2.VideoCapture(video_path)
    #cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)

    # STREAM source
    camera = cv2.VideoCapture(0)

    # Prepare for saving the detected video
    sz = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mpeg')
    fps=20
    
    vout = cv2.VideoWriter()
    vout.open(os.path.join("videos", "res", video), fourcc, fps, sz, True)

    while True:
        res, frame = camera.read()

        if not res:
            break

        image = detect_image(frame, yolo, all_classes, bad_guy, forbbiden_place)
        #cv2.imshow("detection", image)

        # Save the video frame by frame
        vout.write(image)

        if cv2.waitKey(1) & 0xff == 27:
                break

    vout.release()
    camera.release()
    print("done")
    


# In[19]:


yolo = YOLO(0.25, 0.5)
file = 'data/coco_classes.txt'
all_classes = get_classes(file)


# ### Detecting Images

# In[35]:


# DEFINE MISSION:
# choose objects from classes list: data/coco_classes.txt
bad_guy="cat"
forbidden_place="toilet"

with open("data/coco_classes.txt") as f:
    if bad_guy and forbidden_place in f.read():
        print("Objects are valid")
    else:
        print("CHANGE OBJECT NAMES")


# In[20]:


f = 'oscar_toilet.jpg'
path = 'images/'+f
image = cv2.imread(path)
image = detect_image(image, yolo, all_classes,bad_guy, forbidden_place)
cv2.imwrite('images/res/' + f, image)


# # Detecting on Video

# In[36]:


# # detect videos one at a time in videos/test folder    

video = 'cat_flushing2.mp4'
detect_video(video, yolo, all_classes, bad_guy, forbidden_place)


# In[ ]:





# In[ ]:




