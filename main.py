import tensorflow as tf
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4
import cv2
import numpy as np
import time
from threading import Thread, active_count
from collections import deque
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


def detected_photo(boxes, scores, classes, detections, image, image_all, time_detection):
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
            if class_idx == 2 or class_idx == 0 or class_idx == 7:         # show bounding box only to the "car" class
                flag_detection = True
                #### Draw a rectangle ##################
                # convert from tf.Tensor to numpy
                cv2.rectangle(image_cv, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), thickness=2)
                # Add detection text to the prediction
                text = CLASSES[class_idx] + ': {0:.2f}'.format(score)
                cv2.putText(image_cv, text, (int(xmin), int(ymin) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if flag_detection:
        #now_time = time.gmtime()
        #str_time = f'{now_time.tm_mon}_{now_time.tm_mday}_{now_time.tm_hour}_{now_time.tm_min}_{now_time.tm_sec}'
        cv2.imwrite(f'photos/{time_detection}.jpg', image_cv*255)
    # return image_cv


def proccess_frame(model, image_all):
    while len(memory) > 0:
        #images = resize_image(photo)
        print('Размер стека: ', len(memory))
        photo, time_detection = memory.popleft()
        images = resize_image(photo)
        boxes, scores, classes, detections = model.predict(images)
        detected_photo(boxes, scores, classes, detections, images[0], image_all, time_detection)

# Функция вычисления хэша


def CalcImageHash(FileName):
    # image = cv2.imread(FileName)  # Прочитаем картинку
    image_h = FileName
    size_h = 16
    resized = cv2.resize(image_h, (size_h, size_h), interpolation=cv2.INTER_AREA)  # Уменьшим картинку
    gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)  # Переведем в черно-белый формат
    avg = gray_image.mean()  # Среднее значение пикселя
    ret, threshold_image = cv2.threshold(gray_image, avg, 255, 0)  # Бинаризация по порогу

    # Рассчитаем хэш
    _hash = ""
    for x in range(size_h):
        for y in range(size_h):
            val = threshold_image[x, y]
            if val == 255:
                _hash = _hash + "1"
            else:
                _hash = _hash + "0"

    return _hash


def compare_hash(hash1, hash2):
    ln = len(hash1)
    i = 0
    count = 0
    while i < ln:
        if hash1[i] != hash2[i]:
            count = count + 1
        i = i + 1
    return count


def car_detection_video():
    camera = cv2.VideoCapture('rtsp://admin:admin@10.64.130.10:554/h264')
    frame_width_det = (camera.get(cv2.CAP_PROP_FRAME_WIDTH))  # Получаем размер исходного видео
    frame_height_det = (camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dim = (WIDTH, HEIGHT)

    name_window = 'IP Camera - 10.64.130.4'
    # write resulted frames to file
    #out = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'MJPG'), 30, (WIDTH ,HEIGHT))
    success, image = camera.read()
    image = image[int(frame_height_det)//4:int(frame_height_det), 0:int(frame_width_det)]
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    image_tmp = image
    while True:
        # function extract frames
        for __ in range(5):
           _, _ = camera.read()
        success, image2 = camera.read()
        image2_all = image2
        image2 = image2[int(frame_height_det) // 4:int(frame_height_det), 0:int(frame_width_det)]
        image2 = cv2.resize(image2, dim, interpolation=cv2.INTER_AREA)
        diffimg = cv2.absdiff(image, image2)
        d_s = cv2.sumElems(diffimg)
        d = (d_s[0] + d_s[1] + d_s[2]) / (WIDTH * HEIGHT)
        image = image2
        if d > 7:
            hash1 = CalcImageHash(image)
            hash2 = CalcImageHash(image_tmp)
            c_h = compare_hash(hash1, hash2)
            if c_h > 12:
                image_tmp = image
                cv2.namedWindow(name_window, 0)
                cv2.imshow(name_window, image2_all)
                cv2.resizeWindow(name_window, int(frame_width_det) // 2,
                                 int(frame_height_det) // 2)

                memory.append([tf.convert_to_tensor(image), int(time.time() * 1000)])
                print('Стек ADD:', len(memory))
                print('Процессов :', active_count())
                if active_count() == 1:
                    Thread(target=proccess_frame, args=(model, image2_all)).start()
                # Thread(target=proccess_frame,
                #        args=(tf.convert_to_tensor(image), model, image2_all, int(time.time() * 1000))).start()

                cv2.namedWindow(name_window, 0)
                cv2.imshow(name_window, image2_all)
                cv2.resizeWindow(name_window, int(frame_width_det) // 2,
                                 int(frame_height_det) // 2)  # Устанавливаем размер окна вывода
        if cv2.waitKey(2) == 27:  # Выход по ESC
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()

#######   main   ####################################################################

if __name__ == "__main__":

    WIDTH, HEIGHT = 512, 512
    """
    width=608 height=608 in cfg: 65.7% mAP@0.5 (43.5% AP@0.5:0.95) - 34(R) FPS / 62(V) FPS - 128.5 BFlops
    width=512 height=512 in cfg: 64.9% mAP@0.5 (43.0% AP@0.5:0.95) - 45(R) FPS / 83(V) FPS - 91.1 BFlops
    width=416 height=416 in cfg: 62.8% mAP@0.5 (41.2% AP@0.5:0.95) - 55(R) FPS / 96(V) FPS - 60.1 BFlops
    width=320 height=320 in cfg: 60% mAP@0.5 ( 38% AP@0.5:0.95) - 63(R) FPS / 123(V) FPS - 35.5 BFlop
    """

####   Detect Cars on a video ######
    model = trained_yolov4_model()
    memory = deque()
    car_detection_video()
