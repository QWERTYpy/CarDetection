# By Lihi Gur-Arie, 27.4.21
# https://lindevs.com/yolov4-object-detection-using-tensorflow-2/
# https://github.com/sicara/tf2-yolov4/blob/master/notebooks/YoloV4_Dectection_Example.ipynb

import tensorflow as tf
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4
import cv2
import numpy as np
import time
from threading import Thread

#################################################

def load_image_to_tensor(file):
    # load image
    image = tf.io.read_file(file)
    # detect format (JPEG, PNG, BMP, or GIF) and converts to Tensor:
    image = tf.io.decode_image(image)
    return image

def resize_image(image):
    # Resize the output_image:
    image = tf.image.resize(image, (HEIGHT, WIDTH))
    # Add a batch dim:
    images = tf.expand_dims(image, axis=0)/255
    return images

def get_image_from_plot():
    # crates a numpy array from the output_image of the plot\figure
    canvas = FigureCanvasAgg(Figure())
    canvas.draw()
    return np.fromstring(canvas.tostring_rgb(), dtype='uint8')

def trained_yolov4_model():
    # load trained yolov4 model
    model = YOLOv4(
        input_shape=(HEIGHT, WIDTH, 3),
        anchors=YOLOV4_ANCHORS,
        num_classes=80,
        training=False,
        yolo_max_boxes=20,
        yolo_iou_threshold=0.5,
        yolo_score_threshold=0.73,
    )
    model.load_weights('yolov4.h5')
    return model

def detected_photo(boxes, scores, classes, detections,image, image_all, time_detection):
    boxes = (boxes[0] * [WIDTH, HEIGHT, WIDTH, HEIGHT]).astype(int)
    scores = scores[0]
    classes = classes[0].astype(int)
    detections = detections[0]

    CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
        'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    ########################################################################


    image_cv = image.numpy()
    flag_detection = False
    for (xmin, ymin, xmax, ymax), score, class_idx in zip(boxes, scores, classes):

        if score > 0:
        #     if class_idx == 2:
        #         print('Обнаружена машина')
        #     if class_idx == 0:
        #         print('Обнаружен человек')
        #     if class_idx == 7:
        #         print('Обнаружен грузовик')
    #         if class_idx == 2 or class_idx == 0 or class_idx == 7:
    #             flag_detection = True
    # if flag_detection:
    #     now_time = time.gmtime()
    #     str_time = f'{now_time.tm_mon}_{now_time.tm_mday}_{now_time.tm_hour}_{now_time.tm_min}_{now_time.tm_sec}'
    #     cv2.imwrite(f'photos/{time_detection}.jpg', image_all)
        #print('detection')
            if class_idx == 2 or class_idx == 0 or class_idx == 7:         # show bounding box only to the "car" class
                flag_detection = True
                #### Draw a rectangle ##################
                # convert from tf.Tensor to numpy
                cv2.rectangle(image_cv, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,255,0), thickness= 2)
                # Add detection text to the prediction
                text = CLASSES[class_idx] + ': {0:.2f}'.format(score)
                cv2.putText(image_cv, text, (int(xmin), int(ymin) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if flag_detection:
    #now_time = time.gmtime()
    #str_time = f'{now_time.tm_mon}_{now_time.tm_mday}_{now_time.tm_hour}_{now_time.tm_min}_{now_time.tm_sec}'
        cv2.imwrite(f'photos/{time_detection}.jpg', image_cv*255)

    # return image_cv
    #return class_idx

def proccess_frame(photo, model, image_all, time_detection):
    #print('frame_start')
    images = resize_image(photo)
    #images = photo
    boxes, scores, classes, detections = model.predict(images)
    detected_photo(boxes, scores, classes, detections, images[0], image_all, time_detection)
    # print(boxes, scores, classes, detections)
    #return True #result_img

# def Car_detection_single_photo(input_photo):
#     my_image = load_image_to_tensor(input_photo)
#     yolo_model = trained_yolov4_model()
#     image = proccess_frame(my_image, yolo_model)
#     image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert from BGR to RGB
#     return image

# def Car_detection_video(input_video_name, output_video_name, frames_to_save = 50):

# def save_detection_photo(image_crop, image_all):
#     result_img = proccess_frame(tf.convert_to_tensor(image_crop), model)  # tag cars on the frame
    # if result_img == 2 or result_img == 0 or result_img == 7:
    #     now_time = time.gmtime()
    #     str_time = f'{now_time.tm_mon}_{now_time.tm_mday}_{now_time.tm_hour}_{now_time.tm_min}_{now_time.tm_sec}'
    #     cv2.imwrite(f'photos/{int(time.time()*1000)}_{result_img}.jpg', image_all)

# Функция вычисления хэша
def CalcImageHash(FileName):
    #image = cv2.imread(FileName)  # Прочитаем картинку
    image_h = FileName
    resized = cv2.resize(image_h, (32, 32), interpolation=cv2.INTER_AREA)  # Уменьшим картинку
    gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)  # Переведем в черно-белый формат
    avg = gray_image.mean()  # Среднее значение пикселя
    ret, threshold_image = cv2.threshold(gray_image, avg, 255, 0)  # Бинаризация по порогу

    # Рассчитаем хэш
    _hash = ""
    for x in range(32):
        for y in range(32):
            val = threshold_image[x, y]
            if val == 255:
                _hash = _hash + "1"
            else:
                _hash = _hash + "0"

    return _hash


def CompareHash(hash1, hash2):
    l = len(hash1)
    i = 0
    count = 0
    while i < l:
        if hash1[i] != hash2[i]:
            count = count + 1
        i = i + 1
    return count

def Car_detection_video():
    # load trained yolov4 model
    # model = trained_yolov4_model()

    camera = cv2.VideoCapture('rtsp://admin:admin@10.64.130.10:554/h264')
    # camera.set(15, 0.1)
    frame_width_det = (camera.get(cv2.CAP_PROP_FRAME_WIDTH))  # Получаем размер исходного видео
    frame_height_det = (camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dim = (WIDTH, HEIGHT)

    name_window = 'IP Camera - 10.64.130.4'

    # load video
    # my_video = cv2.VideoCapture(input_video_name)

    # write resulted frames to file
    #out = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'MJPG'), 30, (WIDTH ,HEIGHT))

    # success = 1
    # i = 0
    # while success and i < frames_to_save:                                 # While there are more frames in the video
    success, image = camera.read()
    image = image[int(frame_height_det)//4:int(frame_height_det), 0:int(frame_width_det)]
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    image_tmp = image
    while True:
        # function extract frames
        for __ in range(5):
           _, _ = camera.read()
        #success, image = camera.retrieve()                                  # extract a frame
        # time.sleep(0.2)
        success, image2 = camera.read()
        image2_all = image2
        image2 = image2[int(frame_height_det) // 4:int(frame_height_det), 0:int(frame_width_det)]
        image2 = cv2.resize(image2, dim, interpolation=cv2.INTER_AREA)
        diffimg = cv2.absdiff(image, image2)
        d_s = cv2.sumElems(diffimg)
        d = (d_s[0] + d_s[1] + d_s[2]) / (WIDTH * HEIGHT)
        image = image2
        #print('Активность :', d)
        #image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        if d > 7:
            # diffimg = cv2.absdiff(image, image_tmp)
            # d_s = cv2.sumElems(diffimg)
            # d = (d_s[0] + d_s[1] + d_s[2]) / (WIDTH * HEIGHT)
            # #print('Дифф :', d)
            hash1 = CalcImageHash(image)
            hash2 = CalcImageHash(image_tmp)
            c_h = CompareHash(hash1, hash2)
            #print(c_h)
            if c_h > 48:
                print('Ротация')
                image_tmp = image
                cv2.namedWindow(name_window, 0)
                cv2.imshow(name_window, image2_all)
                cv2.resizeWindow(name_window, int(frame_width_det) // 2,
                                 int(frame_height_det) // 2)
                Thread(target=proccess_frame,
                       args=(tf.convert_to_tensor(image), model, image2_all, int(time.time() * 1000))).start()
            d=1
            if d > 15:
                #print('Ротация')
                image_tmp = image

                Thread(target= proccess_frame, args=(tf.convert_to_tensor(image), model, image2_all, int(time.time() * 1000))).start()
                # th.start()
            # result_img = proccess_frame(tf.convert_to_tensor(image), model)   # tag cars on the frame
            #result_img = image
                cv2.namedWindow(name_window, 0)
                cv2.imshow(name_window, image2_all)
                cv2.resizeWindow(name_window, int(frame_width_det) // 2,
                                 int(frame_height_det) // 2)  # Устанавливаем размер окна вывода
            # out.write((result_img*255).astype('uint8'))                                             # write resulted frame to the video file
            # i = i + 1
            # print(i)
        if cv2.waitKey(2) == 27:  # Выход по ESC
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()
    # out.release()                                                         # Close the video writer

#######   main   ####################################################################

if __name__ == "__main__":

    WIDTH, HEIGHT = 1024, 768

    ####    Detect Cars on a single photo ####

    # output_image = Car_detection_single_photo(input_photo ='photos/test3.jpg')
    #
    # # Show resulted photo
    # cv2.imshow('output_image', output_image)
    # cv2.waitKey(3000)
    # cv2.destroyAllWindows()
    #
    # # Save photo
    # cv2.imwrite('photos/test3_object_detected.jpg', output_image*255)

####   Detect Cars on a video and save ######
    # Car_detection_video(input_video_name='photos/car_chase_01.mp4', output_video_name ='photos/delete.avi', frames_to_save = 20)
    model = trained_yolov4_model()
    Car_detection_video()
