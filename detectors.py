from ultralytics import YOLO
from ultralytics import RTDETR
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

class YoloDetector:
    """
    Детектор людей с использованием стандартного YOLO.

    Attributes:
        model_path (str): Путь к файлу модели YOLO.
        preprocessing_func (callable, optional): Функция предобработки изображения.
        device (str): Устройство для вычислений.
        model (YOLO): Загруженная модель YOLO.
    """

    def __init__(self, model_path, preprocessing_func=None, device="cpu"):
        """
        Инициализирует детектор на базе YOLO.

        Args:
            model_path (str): Путь к модели YOLO, если она не скачана, но существует, 
                то она автоматически скачается (например, 'yolo12n.pt').
            preprocessing_func (callable, optional): Функция предобработки кадров.
                По умолчанию None.
            device (str, optional): Устройство для вычислений ("cuda" или "cpu").
                По умолчанию "cpu".
        """
        self.model_path = model_path
        self.preprocessing_func = preprocessing_func
        self.device = device

        print(f"Загрузка модели {model_path} (Чистый YOLO)...")
        self.model = YOLO(model_path)

    def detect(self, frame):
        """
        Детектирует людей на кадре с использованием YOLO.

        Выполняет детекцию на полном изображении без разбиения.

        Args:
            frame (numpy.ndarray): Кадр видео для обработки.

        Returns:
            list: Список словарей с обнаруженными людьми. Каждый словарь содержит:
                - 'bbox' (tuple): Координаты ограничивающего прямоугольника
                  в формате (x1, y1, x2, y2).
                - 'confidence' (float): Уверенность детекции (0-1).
        """
        detections = []

        if self.preprocessing_func:
            frame = self.preprocessing_func(frame)

        # Выполняем детекцию
        results = self.model(frame, device=self.device, verbose=False)

        # Обрабатываем результаты
        for result in results:
            boxes = result.boxes

            for box in boxes:
                # Получаем класс объекта (0 - человек в COCO dataset)
                cls = int(box.cls[0])

                # Фильтруем только людей
                if cls == 0:  # класс "person"
                    # Получаем координаты bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Получаем уверенность
                    confidence = float(box.conf[0])

                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence
                    })

        return detections
    
    
class YoloSahiDetector:
    """
    Детектор людей с использованием YOLO и SAHI для улучшенной детекции.

    Attributes:
        model_path (str): Путь к файлу модели YOLO.
        slice_height (int): Высота фрагмента изображения для обработки.
        slice_width (int): Ширина фрагмента изображения для обработки.
        overlap_height_ratio (float): Коэффициент перекрытия фрагментов по высоте.
        overlap_width_ratio (float): Коэффициент перекрытия фрагментов по ширине.
        preprocessing_func (callable, optional): Функция предобработки изображения.
        model (AutoDetectionModel): Загруженная модель детекции.
    """

    def __init__(self, model_path,
                 slice_height=512, slice_width=512,
                 overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                 preprocessing_func=None,
                 device="cpu"):
        """
        Инициализирует детектор YOLO с поддержкой SAHI.

        Args:
            model_path (str): Путь к модели YOLO, если она не скачана, но существует, 
                то она автоматически скачается (например, 'yolo12n.pt').
            slice_height (int, optional): Высота фрагмента для SAHI.
                По умолчанию 512.
            slice_width (int, optional): Ширина фрагмента для SAHI.
                По умолчанию 512.
            overlap_height_ratio (float, optional): Процент перекрытия по высоте.
                По умолчанию 0.2.
            overlap_width_ratio (float, optional): Процент перекрытия по ширине.
                По умолчанию 0.2.
            preprocessing_func (callable, optional): Функция предобработки кадров.
                По умолчанию None.
            device (str, optional): Устройство для вычислений ("cuda" или "cpu").
                По умолчанию "cuda".
        """
        self.model_path = model_path
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.preprocessing_func = preprocessing_func

        print(f"Загрузка модели {model_path}...")
        self.model = AutoDetectionModel.from_pretrained(
            model_type='ultralytics',
            model_path=model_path,
            device=device
        )

    def detect(self, frame):
        """
        Детектирует людей на кадре с использованием SAHI.

        Метод разбивает изображение на перекрывающиеся фрагменты,
        выполняет детекцию на каждом и объединяет результаты.

        Args:
            frame (numpy.ndarray): Кадр видео для обработки.

        Returns:
            list: Список словарей с обнаруженными людьми. Каждый словарь содержит:
                - 'bbox' (tuple): Координаты ограничивающего прямоугольника
                  в формате (x1, y1, x2, y2).
                - 'confidence' (float): Уверенность детекции (0-1).
        """
        detections = []

        if self.preprocessing_func:
            frame = self.preprocessing_func(frame)

        # Детекция с разбиением на фрагменты
        result = get_sliced_prediction(
            frame,
            self.model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            verbose=0
        )

        for object_prediction in result.object_prediction_list:
            # Фильтруем только людей (класс 'person' или ID = 0)
            if (object_prediction.category.name == 'person' or
                    object_prediction.category.id == 0):
                bbox = object_prediction.bbox
                x1 = int(bbox.minx)
                y1 = int(bbox.miny)
                x2 = int(bbox.maxx)
                y2 = int(bbox.maxy)
                confidence = object_prediction.score.value

                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence
                })

        return detections

    
class RTDETRDetector:
    """
    Детектор людей с использованием RTDETR.

    Attributes:
        model_path (str): Путь к файлу модели RTDETR.
        preprocessing_func (callable, optional): Функция предобработки изображения.
        device (str): Устройство для вычислений.
        model (RTDETR): Загруженная модель RTDETR.
    """

    def __init__(self, model_path = "rtdetr-x.pt", preprocessing_func=None, device="cpu"):
        """
        Инициализирует детектор на базе RTDETR.

        Args:
            model_path (str): Путь к модели RTDETR, если она не скачана, но существует, 
                то она автоматически скачается (rtdetr-l.pt/rtdetr-x.pt).
            preprocessing_func (callable, optional): Функция предобработки кадров.
                По умолчанию None.
            device (str, optional): Устройство для вычислений ("cuda" или "cpu").
                По умолчанию "cpu".
        """
        self.preprocessing_func = preprocessing_func
        self.device = device
        self.model = RTDETR(model_path)

    def detect(self, frame):
        """
        Детектирует людей на кадре с использованием RTDETR.

        Выполняет детекцию на полном изображении без разбиения.

        Args:
            frame (numpy.ndarray): Кадр видео для обработки.

        Returns:
            list: Список словарей с обнаруженными людьми. Каждый словарь содержит:
                - 'bbox' (tuple): Координаты ограничивающего прямоугольника
                  в формате (x1, y1, x2, y2).
                - 'confidence' (float): Уверенность детекции (0-1).
        """
        detections = []

        if self.preprocessing_func:
            frame = self.preprocessing_func(frame)

        # Выполняем детекцию
        results = self.model(frame, device=self.device, verbose=False)

        # Обрабатываем результаты
        for result in results:
            boxes = result.boxes

            for box in boxes:
                # Получаем класс объекта (0 - человек в COCO dataset)
                cls = int(box.cls[0])

                # Фильтруем только людей
                if cls == 0:  # класс "person"
                    # Получаем координаты bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Получаем уверенность
                    confidence = float(box.conf[0])

                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence
                    })

        return detections
    
    
class RTDETRSahiDetector:
    """
    Детектор людей с использованием RTDETR и SAHI для улучшенной детекции.

    Attributes:
        model_path (str): Путь к файлу модели RTDETR.
        slice_height (int): Высота фрагмента изображения для обработки.
        slice_width (int): Ширина фрагмента изображения для обработки.
        overlap_height_ratio (float): Коэффициент перекрытия фрагментов по высоте.
        overlap_width_ratio (float): Коэффициент перекрытия фрагментов по ширине.
        preprocessing_func (callable, optional): Функция предобработки изображения.
        model (AutoDetectionModel): Загруженная модель детекции.
    """

    def __init__(self, model_path = "rtdetr-x.pt",
                 slice_height=512, slice_width=512,
                 overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                 preprocessing_func=None,
                 device="cpu"):
        """
        Инициализирует детектор RTDETR с поддержкой SAHI.

        Args:
            model_path (str): Путь к модели RTDETR, если она не скачана, но существует, 
                то она автоматически скачается (например, 'rtdetr-x.pt').
            slice_height (int, optional): Высота фрагмента для SAHI.
                По умолчанию 512.
            slice_width (int, optional): Ширина фрагмента для SAHI.
                По умолчанию 512.
            overlap_height_ratio (float, optional): Процент перекрытия по высоте.
                По умолчанию 0.2.
            overlap_width_ratio (float, optional): Процент перекрытия по ширине.
                По умолчанию 0.2.
            preprocessing_func (callable, optional): Функция предобработки кадров.
                По умолчанию None.
            device (str, optional): Устройство для вычислений ("cuda" или "cpu").
                По умолчанию "cpu".
        """
        self.model_path = model_path
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.preprocessing_func = preprocessing_func

        print(f"Загрузка модели {model_path}...")
        self.model = AutoDetectionModel.from_pretrained(
            model_type='ultralytics',
            model_path=model_path,
            device=device
        )

    def detect(self, frame):
        """
        Детектирует людей на кадре с использованием SAHI.

        Метод разбивает изображение на перекрывающиеся фрагменты,
        выполняет детекцию на каждом и объединяет результаты.

        Args:
            frame (numpy.ndarray): Кадр видео для обработки.

        Returns:
            list: Список словарей с обнаруженными людьми. Каждый словарь содержит:
                - 'bbox' (tuple): Координаты ограничивающего прямоугольника
                  в формате (x1, y1, x2, y2).
                - 'confidence' (float): Уверенность детекции (0-1).
        """
        detections = []

        if self.preprocessing_func:
            frame = self.preprocessing_func(frame)

        # Детекция с разбиением на фрагменты
        result = get_sliced_prediction(
            frame,
            self.model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            verbose=0
        )

        for object_prediction in result.object_prediction_list:
            # Фильтруем только людей (класс 'person' или ID = 0)
            if (object_prediction.category.name == 'person' or
                    object_prediction.category.id == 0):
                bbox = object_prediction.bbox
                x1 = int(bbox.minx)
                y1 = int(bbox.miny)
                x2 = int(bbox.maxx)
                y2 = int(bbox.maxy)
                confidence = object_prediction.score.value

                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence
                })

        return detections