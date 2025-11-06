import argparse
from pathlib import Path

import torch

import detectors
from preprocessors import apply_clahe
import video_writers

def parse_arguments():
    """
    Парсинг аргументов командной строки.

    Returns:
        argparse.Namespace: Объект с распарсенными аргументами, содержащий:
            - input (str): Путь к входному видеофайлу.
            - output (str): Путь к выходному видеофайлу.
            - confidence (float): Порог уверенности для детекции.
    """
    parser = argparse.ArgumentParser(
        description="Детекция людей на видео с использованием YOLO"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Путь к входному видеофайлу"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.mp4",
        help="Путь к выходному видеофайлу (по умолчанию: output.mp4)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Порог уверенности для детекции (по умолчанию: 0.5)"
    )

    return parser.parse_args()


def validate_paths(input_path, output_path):
    """
    Валидация входных и выходных путей.

    Проверяет существование входного файла и создаёт директорию
    для выходного файла при необходимости.

    Args:
        input_path (str): Путь к входному файлу.
        output_path (str): Путь к выходному файлу.

    Returns:
        tuple: Кортеж из Path объектов (input_path, output_path).

    Raises:
        FileNotFoundError: Если входной файл не существует.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Входной файл не найден: {input_path}")

    # Создаём директорию для выходного файла, если её нет
    output_path.parent.mkdir(parents=True, exist_ok=True)

    return input_path, output_path




def main():
    """
    Главная функция программы.

    Выполняет следующие шаги:
    1. Парсинг аргументов командной строки.
    2. Валидация входных и выходных путей.
    3. Определение доступного устройства (CUDA/CPU).
    4. Инициализация детектора RT-DETR.
    5. Обработка видео с детекцией людей.

    Returns:
        int: Код возврата (0 - успех, 1 - ошибка).
    """
    try:
        # Парсинг аргументов
        args = parse_arguments()

        # Валидация путей
        input_path, output_path = validate_paths(args.input, args.output)

        print(f"Входной файл: {input_path}")
        print(f"Выходной файл: {output_path}")
        print(f"Порог уверенности: {args.confidence}")
        print("-" * 50)

        # Определение устройства для вычислений
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        confidence = args.confidence
        
        # Инициализация модели RT-DETR + SAHI
        model = detectors.RTDETRSahiDetector(
            'rtdetr-x.pt',
            slice_height=800,
            slice_width=800,
            overlap_height_ratio=0.3,
            overlap_width_ratio=0.3,
            preprocessing_func=apply_clahe,
            device=device
        )
        # Обработка видео
        video_writers.draw_detections(
            input_path,
            output_path,
            model,
            confidence
        )
        
        
        # Пример применения нескольких моделей
        """
        model1 = detectors.YoloDetector(
            'yolo12l.pt',
            device=device
        )

        model2 = detectors.YoloSahiDetector(
            'yolo12n.pt',
            slice_height=800,
            slice_width=800,
            overlap_height_ratio=0.3,
            overlap_width_ratio=0.3,
            preprocessing_func=apply_clahe,
            device=device
        )

        model3 = detectors.RTDETRDetector(
            model_path="rtdetr-l.pt",
            preprocessing_func=apply_clahe,
            device=device
        )

        video_writers.draw_multi_detections(
            input_path,
            output_path,
            [model1, model2, model3],
            confidence
        )
        """

        print(f"\nОбработка завершена! Результат сохранён в: {output_path}")

    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        return 1
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())