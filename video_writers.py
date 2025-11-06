import cv2


def draw_detections(input_video_path, output_video_path, model,
                    confidence_threshold=0.5):
    """
    Детектирует людей на видео и отрисовывает результаты.

    Обрабатывает видеофайл покадрово, выполняет детекцию людей
    с помощью заданной модели и сохраняет результат с нанесёнными
    bounding box и метками.

    Args:
        input_video_path (str): Путь к исходному видео.
        output_video_path (str): Путь для сохранения обработанного видео.
        model: Объект-детектор с методом detect() (например, YoloSahiDetector).
        confidence_threshold (float, optional): Минимальный порог уверенности
            для отображения детекции. По умолчанию 0.5.

    Returns:
        None
    """
    # Открываем видео
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео {input_video_path}")
        return

    # Получаем параметры видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nПараметры видео: {width}x{height}, {fps} FPS, "
          f"{total_frames} кадров")

    # Создаем VideoWriter для сохранения результата
    fourcc = cv2.VideoWriter_fourcc(*'h264')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0

    print("\nНачинаем обработку видео...")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        input_frame = frame
        frame_count += 1

        # Получаем детекции от модели
        detections = model.detect(input_frame)

        # Отрисовываем каждую детекцию
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']

            if confidence >= confidence_threshold:
                # Рисуем bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                # Создаем текст с уверенностью
                label = f'Person: {confidence:.2f}'

                # Размер текста для фона
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                )

                # Рисуем фон для текста
                cv2.rectangle(
                    frame,
                    (x1, y1 - text_height - 8),
                    (x1 + text_width, y1),
                    (0, 255, 0),
                    -1
                )

                # Рисуем текст
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 0),
                    1
                )

        # Записываем кадр
        out.write(frame)

        # Показываем прогресс
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Обработано кадров: {frame_count}/{total_frames} "
                  f"({progress:.1f}%)")

    # Освобождаем ресурсы
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\n✓ Готово! Видео сохранено в {output_video_path}")
    
    
    
def draw_multi_detections(input_video_path, output_video_path, models,
                                confidence_threshold=0.5):
    """
    Детектирует людей на видео с помощью нескольких моделей и отрисовывает результаты.

    Функция обрабатывает видео покадрово, применяя несколько моделей детекции
    одновременно. Каждая модель отображается своим цветом для визуального
    различения результатов.

    Args:
        input_video_path (str): Путь к исходному видеофайлу.
        output_video_path (str): Путь для сохранения обработанного видео.
        models (list): Список объектов-детекторов (экземпляры YoloSahiDetector
            или любые объекты с методом detect()).
        confidence_threshold (float, optional): Минимальный порог уверенности
            для отображения детекции. По умолчанию 0.5.

    Returns:
        None

    Note:
        Каждая модель визуализируется своим цветом. Поддерживается до 9 моделей
        с уникальными цветами, после чего цвета повторяются.
        Для отличия моделей рисуется bounding box с отличными отступами 
        для каждой модели. 
    """
    # Цветовая палитра для различных моделей
    colors = [
        (255, 0, 0),      # Красный
        (0, 255, 0),      # Зеленый
        (0, 0, 255),      # Синий
        (255, 255, 0),    # Желтый
        (255, 0, 255),    # Пурпурный
        (0, 255, 255),    # Голубой
        (255, 165, 0),    # Оранжевый
        (128, 0, 128),    # Фиолетовый
        (255, 192, 203)   # Розовый
    ]

    # Открываем видео
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео {input_video_path}")
        return

    # Получаем параметры видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nПараметры видео: {width}x{height}, {fps} FPS, "
          f"{total_frames} кадров")
    print(f"Количество моделей: {len(models)}")

    # Создаем VideoWriter для сохранения результата
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0

    print("\nНачинаем обработку видео...")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        input_frame = frame
        frame_count += 1

        # Обрабатываем каждой моделью
        for model_idx, model in enumerate(models):
            # Получаем цвет для текущей модели
            color = colors[model_idx % len(colors)]

            # Получаем детекции от модели
            detections = model.detect(input_frame)

            # Отрисовываем каждую детекцию
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection['confidence']

                if confidence >= confidence_threshold:
                    # Рисуем bounding box со смещением для различных моделей
                    offset = 2 * model_idx
                    cv2.rectangle(
                        frame,
                        (x1 - offset, y1 - offset),
                        (x2 + model_idx, y2 + offset),
                        color,
                        1
                    )

                    # Создаем текст с уверенностью и названием модели
                    model_name = f'Model{model_idx + 1}'
                    label = f'{model_name}: {confidence:.2f}'

                    # Размер текста для фона
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                    )

                    # Рисуем фон для текста
                    cv2.rectangle(
                        frame,
                        (x1, y1 - text_height - 8),
                        (x1 + text_width, y1),
                        color,
                        -1
                    )

                    # Рисуем текст
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 0),
                        1
                    )

        # Записываем кадр
        out.write(frame)

        # Показываем прогресс каждые 30 кадров
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Обработано кадров: {frame_count}/{total_frames} "
                  f"({progress:.1f}%)")

    # Освобождаем ресурсы
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\n✓ Готово! Видео сохранено в {output_video_path}")