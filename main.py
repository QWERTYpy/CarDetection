import tensorflow as tf
from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4
import cv2
import time
from threading import Thread, active_count
from collections import deque
#################################################


def trained_yolov4_model():
    # Загружаем обученную модель yolov4
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


def detected_photo(boxes, scores, classes, detections, image, time_detection):
    # Ищем на переданной фотографии заданные объекты
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
    return flag_detection, time_detection, image_cv


def save_frame_as_photo(image, time_detection):
    cv2.imwrite(f'photos/{time_detection}.jpg', image)


def proccess_frame(model):
    # Обрабатываем изображения из стека
    while len(memory) > 0:
        print('Размер стека: ', len(memory))
        images, time_detection = memory.popleft()
        images = tf.image.resize(images, (HEIGHT, WIDTH))
        # Add a batch dim:
        images = tf.expand_dims(images, axis=0) / 255

        boxes, scores, classes, detections = model.predict(images)
        flag_detection, time_detection, image_cv = detected_photo(boxes, scores, classes, detections, images[0], time_detection)
        if flag_detection:
            save_frame_as_photo(image_cv*255, time_detection)


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
    # Анализируем видео поток
    camera = cv2.VideoCapture('rtsp://admin:admin@10.64.130.10:554/h264')
    frame_width_det = (camera.get(cv2.CAP_PROP_FRAME_WIDTH))  # Получаем размер исходного видео
    frame_height_det = (camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dim = (WIDTH, HEIGHT)

    name_window = 'IP Camera - 10.64.130.4'
    success, image = camera.read()
    # Обрезаем фрейм для детекции движения в определенной области
    image = image[int(frame_height_det)//4:int(frame_height_det), 0:int(frame_width_det)]
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    image_tmp = image
    while True:
        # Поочередно извлекаем фреймы из потока
        # Пропускаем 5 шт для уменьшения нагрузки и детекции движения
        for __ in range(5):
           _, _ = camera.read()
        success, image2 = camera.read()
        image2_all = image2
        #image2 = image2[int(frame_height_det) // 4:int(frame_height_det), 0:int(frame_width_det)]
        image2 = image2[int(frame_height_det)//4:int(frame_height_det), 0:int(frame_width_det)]
        image2 = cv2.resize(image2, dim, interpolation=cv2.INTER_AREA)
        diffimg = cv2.absdiff(image, image2)
        d_s = cv2.sumElems(diffimg)
        d = (d_s[0] + d_s[1] + d_s[2]) / (WIDTH * HEIGHT)
        image = image2
        # Если активность больше заданного порога
        if d > 7:
            # Сравниваем фреймы на похожесть, чтобы исключить избыточности
            # Данный алгоритм работает коряво, но позволяет отфильтровать часть
            hash1 = CalcImageHash(image)
            hash2 = CalcImageHash(image_tmp)
            c_h = compare_hash(hash1, hash2)
            # Если различие выше порога
            if c_h > 12:
                image_tmp = image
                cv2.namedWindow(name_window, 0)
                cv2.imshow(name_window, image2_all)  # Показываем оригинальный фрейм
                cv2.resizeWindow(name_window, int(frame_width_det) // 2,
                                 int(frame_height_det) // 2)

                memory.append([tf.convert_to_tensor(image), int(time.time() * 1000)])  # Запоминаем его
                print('Стек ADD:', len(memory))  # Выводим количство фреймов в стеке
                if active_count() == 1:  # Если в данные момент запущен только один процесс, то запускаем обработку
                    Thread(target=proccess_frame, args=(model,)).start()

        if cv2.waitKey(2) == 27:  # Выход по ESC
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()

#######   main   ####################################################################

if __name__ == "__main__":

    WIDTH, HEIGHT = int(1920/4), int(768/4)
    """
    width=608 height=608 in cfg: 65.7% mAP@0.5 (43.5% AP@0.5:0.95) - 34(R) FPS / 62(V) FPS - 128.5 BFlops
    width=512 height=512 in cfg: 64.9% mAP@0.5 (43.0% AP@0.5:0.95) - 45(R) FPS / 83(V) FPS - 91.1 BFlops
    width=416 height=416 in cfg: 62.8% mAP@0.5 (41.2% AP@0.5:0.95) - 55(R) FPS / 96(V) FPS - 60.1 BFlops
    width=320 height=320 in cfg: 60% mAP@0.5 ( 38% AP@0.5:0.95) - 63(R) FPS / 123(V) FPS - 35.5 BFlop
    """

####   Detect Cars on a video ######
    model = trained_yolov4_model()  # Загружаем обученную модель
    memory = deque()  # Создаем стек
    car_detection_video()  # Запускаем детектор
