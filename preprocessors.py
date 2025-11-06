import cv2
import numpy as np


def apply_clahe(image):
    """
    Применяет CLAHE (адаптивную гистограммную эквализацию с ограничением контраста).

    CLAHE применяется к каналу L в цветовом пространстве LAB для улучшения
    локального контраста с сохранением цветовой информации.

    Args:
        image (numpy.ndarray): Входное изображение в формате BGR.

    Returns:
        numpy.ndarray: Изображение с применённым CLAHE в формате BGR.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def retinex_ssr(image, sigma=300):
    """
    Применяет алгоритм Single-Scale Retinex (SSR) для улучшения изображения.

    Метод улучшает детали изображения путём удаления эффектов освещения
    с использованием логарифмического преобразования и гауссова размытия.

    Args:
        image (numpy.ndarray): Входное изображение в формате BGR.
        sigma (int, optional): Стандартное отклонение для размытия по Гауссу.
            По умолчанию 300.

    Returns:
        numpy.ndarray: Улучшенное изображение с нормализованными значениями (0-255).
    """
    image = image.astype(np.float32) + 1.0
    gaussian = cv2.GaussianBlur(image, (0, 0), sigma)
    retinex = np.log10(image) - np.log10(gaussian)
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
    return retinex.astype(np.uint8)


def adjust_gamma(image, gamma=1.5):
    """
    Применяет гамма-коррекцию для настройки яркости изображения.

    Гамма-коррекция используется для улучшения тёмных или светлых областей
    изображения с помощью степенного преобразования.

    Args:
        image (numpy.ndarray): Входное изображение.
        gamma (float, optional): Значение гаммы для коррекции.
            Значения < 1 осветляют изображение, значения > 1 затемняют.
            По умолчанию 1.5.

    Returns:
        numpy.ndarray: Изображение с применённой гамма-коррекцией.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def sharpen_image(image):
    """
    Повышает резкость изображения с помощью техники нерезкого маскирования.

    Метод создаёт более резкую версию путём вычитания размытой
    версии изображения из оригинала.

    Args:
        image (numpy.ndarray): Входное изображение.

    Returns:
        numpy.ndarray: Изображение с повышенной резкостью.
    """
    gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
    return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)


def sharpen_kernel(image):
    """
    Повышает резкость изображения с использованием свёрточного ядра.

    Применяет ядро размером 3x3 для усиления краёв и деталей.

    Args:
        image (numpy.ndarray): Входное изображение.

    Returns:
        numpy.ndarray: Изображение с повышенной резкостью.
    """
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


def denoise_bilateral(image):
    """
    Применяет билатеральную фильтрацию для шумоподавления с сохранением краёв.

    Билатеральный фильтр сглаживает изображение, сохраняя края резкими,
    учитывая как пространственные, так и интенсивностные различия.

    Args:
        image (numpy.ndarray): Входное изображение.

    Returns:
        numpy.ndarray: Изображение с подавленным шумом и сохранёнными краями.
    """
    return cv2.bilateralFilter(image, 9, 75, 75)


def denoise_nlm(image):
    """
    Применяет нелокальное усреднение для шумоподавления цветного изображения.

    Метод эффективно удаляет шум, сохраняя детали текстуры.

    Args:
        image (numpy.ndarray): Входное цветное изображение в формате BGR.

    Returns:
        numpy.ndarray: Цветное изображение с подавленным шумом.
    """
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


def denoise_gaussian(image):
    """
    Применяет размытие по Гауссу для лёгкого шумоподавления.

    Использует ядро Гаусса размером 5x5 для сглаживания изображения и уменьшения шума.

    Args:
        image (numpy.ndarray): Входное изображение.

    Returns:
        numpy.ndarray: Сглаженное изображение.
    """
    return cv2.GaussianBlur(image, (5, 5), 0)


def auto_brightness_contrast(image, clip_hist_percent=1):
    """
    Автоматически настраивает яркость и контраст с помощью обрезки гистограммы.

    Метод вычисляет оптимальные значения alpha (контраст) и beta (яркость)
    путём анализа гистограммы изображения и обрезки экстремальных значений.

    Args:
        image (numpy.ndarray): Входное изображение в формате BGR.
        clip_hist_percent (float, optional): Процент гистограммы для обрезки
            с обоих концов. По умолчанию 1.

    Returns:
        numpy.ndarray: Изображение с настроенными яркостью и контрастом.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)