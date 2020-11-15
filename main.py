import cv2
import torch
from PIL import ImageGrab
import numpy as np
from model import CnnEmotions
from mss import mss


dict_indexes = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
dict_colors = {0: (255, 0, 0), 1: (128, 128, 0), 2: (255, 105, 180), 3: (255, 165, 0), 4: (112, 128, 144),
               5: (255, 255, 0), 6: (255, 255, 255)}


em = CnnEmotions()
em.load_state_dict(torch.load("res_model.pth", map_location=torch.device("cpu")))


def view_image(image, name_of_window):
    """
    фцнкция выводит картинку
    :param image: Картинка в формате np.array
    :param name_of_window: Название окна
    :return: none
    """
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# cv2 модель для нахождения лиц в кадре
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def get_faces(image):
    """
    Аозвращает лица из изображения
    :param image: Входное изображение
    :return: Возвращает лица на изображнии в формате (x, y, w, h)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(10, 10)
    )
    # return sorted(faces, key=lambda x: get_square_of_image(x), reverse=True)
    return faces

def to_grey(image):
    """
    делает изображение серым
    :param image: Входное np.array изображение
    :return: серое одноканальное np.array изображение
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def scale(image, res):
    """

    :param image: Входное np.array изображение
    :param res: ширина и высота выходного изобраджения
    :return: изображение np.array формата (res, res, n_channels)
    """
    try:
        return cv2.resize(image, (res, res), interpolation=cv2.INTER_AREA)
    except:
        return np.zeros(res, res, 3)


def get_square_of_image(params):
    """
    Вощвращает площадь изображения
    :param params: параметры выходного изображение
    :return: площадь выходного изображения
    """
    return params[2] * params[3]


def create_borders(image, color, size=20):
    """
    Делает рамку вокруг каждого лица
    :param image: Входное np.array изображение
    :param color: Цвет рамки
    :param size: Ширина рамки
    :return: Лицо, обведенное в рамку заданного цвета
    """
    height, width = image.shape[:-1]
    blank_image = np.zeros((height + size, width + size, 3), np.uint8)
    blank_image[:, :] = color[::-1]
    blank_image[size // 2:size // 2 + height, size // 2: size // 2 + width] = image[::]

    return blank_image


def collect_faces_to_image(image, faces, texts, scalemod=100):
    """
    Делает итоговую картинку для вывода на экран
    :param image: Входное np.array изображение
    :param faces: (x, y, w, h) всех лиц на изображении
    :param texts: Классы лиц по эмоциям
    :param scalemod: Размер изображения лица до рамки
    :return: Изображение размерв (n_faces*face_high, face_width, 3)
    """
    res = []
    i = 0
    color = dict(zip(dict_indexes.values(), dict_indexes.keys()))
    for (x, y, _w, _h) in faces:
        new_face = scale(image[y:y + _h, x:x + _w], scalemod)
        org = (0+1, scalemod-1)
        color_t = dict_colors[color[texts[i]]]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontscale = 0.5
        thickness = 2
        new_face = cv2.putText(new_face, texts[i], org, font,
                            fontscale, color_t[::-1], thickness, cv2.LINE_AA)
        new_face = create_borders(new_face, color_t)

        res.append(new_face)
        i += 1
    if res:
        return np.concatenate(res)
    else:
        return np.zeros((100, 100, 3))


def detect_face_and_draw_rectangle(image, faces, texts=None):
    """
    Определяет лица и обводит их в рамку
    :param image: Входное изображение np.array
    :param faces: (x, y, w, h) всех лиц на изображении
    :param texts:  Текстовые описания всех лиц на фото
    :return: Входное изображение np.array
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    color = (0, 0, 255)
    thickness = 2

    i = 0
    if not texts:
        texts = ["VATman" for _ in range(len(faces))]
    for (x, y, w, h) in faces:
        image = cv2.putText(image, texts[i], (x+w, y+h), font, fontscale, color, thickness, cv2.LINE_AA)
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
        i += 1
    return image


def show_face_with_nn(picture_taker, model, operation):
    """
    Основная функция
    :param picture_taker: метод возращающий np.array картинку
    :param model: нейронная сеть
    :param operation: метод преобразования изображений
    :return: none
    """
    model.train(False)
    while True:
        image = picture_taker()
        faces = get_faces(image)
        faces = faces[:min(len(faces), 30)]
        emotions = []
        for (x, y, w, h) in faces:
            face = scale(to_grey(image[y:y + h, x:x + w]), 48)
            face = torch.tensor([face])
            pred = model.forward(face)
            emotion = dict_indexes[int(pred.argmax(axis=-1)[0])]
            emotions.append(emotion)

        image = operation(image, faces, emotions)
        cv2.imshow('frame', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


# Ширина и высота захваченного изображения
w = np.array(ImageGrab.grab()).shape[0] // 2
h = np.array(ImageGrab.grab()).shape[1] // 2


def image_grabber():
    """
    Запись экрана
    :return: Кадр с экрана np.array
    """
    img = np.array(ImageGrab.grab(bbox=(0, 0, w, h)))[:, :, [2, 1, 0]]
    return cv2.resize(img, (img.shape[1]//4, img.shape[0]//4), interpolation=cv2.INTER_AREA)


def capture_screenshot():
    """
    Запись экрана c большей частотой кадров
    :return: Кадр с экрана np.array
    """
    with mss() as sct:
        monitor = {'left': 0, 'top': 0, 'width': sct.monitors[1]['width']//2, 'height': sct.monitors[1]['height']}
        sct_img = sct.grab(monitor)
        img = np.array(sct_img)[:, :, [0, 1, 2]]
        return cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4), interpolation=cv2.INTER_AREA)

# def frontCamera():
#     """
#     Запись с камеры
#     :return: кадр с камеры np.array
#     """
#     return vid.read()[-1]

# вызов основной функции
show_face_with_nn(capture_screenshot, em, collect_faces_to_image)

# vid = cv2.VideoCapture(0)
# show_face_with_nn(frontCamera, em, collect_faces_to_image)
# vid.release()



