#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пример использования ONNX модели для инференса
Автор: AI Assistant
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import torch

class YOLOv8ONNXInference:
    """Класс для инференса YOLOv8 ONNX модели"""
    
    def __init__(self, onnx_path, conf_threshold=0.5, iou_threshold=0.45):
        """
        Инициализация ONNX модели
        
        Args:
            onnx_path: путь к ONNX файлу
            conf_threshold: порог уверенности
            iou_threshold: порог IoU для NMS
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Загружаем ONNX модель
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
        
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Получаем информацию о входе и выходе
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"✅ ONNX модель загружена: {onnx_path}")
        print(f"   Вход: {self.input_name}, форма: {self.input_shape}")
        print(f"   Выходы: {self.output_names}")
        print(f"   Провайдеры: {self.session.get_providers()}")
        
        # Классы
        self.class_names = ['clean', 'dirty', 'dented', 'scratched', 'broken']
        self.class_colors = [
            (0, 255, 0),      # clean - зеленый
            (0, 165, 255),    # dirty - оранжевый
            (0, 0, 255),      # dented - красный
            (255, 0, 0),      # scratched - синий
            (128, 0, 128),    # broken - фиолетовый
        ]
    
    def preprocess_image(self, image_path, target_size=640):
        """
        Предобработка изображения для YOLOv8
        
        Args:
            image_path: путь к изображению
            target_size: размер для ресайза
            
        Returns:
            preprocessed_image: предобработанное изображение
            original_image: оригинальное изображение
            scale_factor: коэффициент масштабирования
        """
        # Загружаем изображение
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        original_image = image.copy()
        h, w = image.shape[:2]
        
        # Ресайз с сохранением пропорций
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        image = cv2.resize(image, (new_w, new_h))
        
        # Добавляем паддинг до target_size x target_size
        pad_h = target_size - new_h
        pad_w = target_size - new_w
        
        image = cv2.copyMakeBorder(
            image, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # Нормализация и изменение порядка каналов
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        image = np.expand_dims(image, axis=0)   # Добавляем batch dimension
        
        return image, original_image, scale
    
    def postprocess_outputs(self, outputs, scale_factor, original_shape):
        """
        Постобработка выходов модели
        
        Args:
            outputs: выходы ONNX модели
            scale_factor: коэффициент масштабирования
            original_shape: форма оригинального изображения
            
        Returns:
            boxes: координаты bbox
            scores: уверенность
            class_ids: ID классов
        """
        # YOLOv8 выдает один выход с формой [1, 84, 8400]
        # где 84 = 4 (bbox) + 80 (классы COCO) - но у нас 5 классов
        predictions = outputs[0]  # [1, 84, 8400]
        predictions = np.transpose(predictions, (0, 2, 1))  # [1, 8400, 84]
        
        # Извлекаем bbox и scores
        boxes = predictions[0, :, :4]  # [8400, 4] - x_center, y_center, width, height
        scores = predictions[0, :, 4:]  # [8400, 80] - scores для всех классов
        
        # Фильтруем по порогу уверенности
        max_scores = np.max(scores, axis=1)
        valid_indices = max_scores > self.conf_threshold
        
        if not np.any(valid_indices):
            return np.array([]), np.array([]), np.array([])
        
        boxes = boxes[valid_indices]
        scores = scores[valid_indices]
        max_scores = max_scores[valid_indices]
        
        # Получаем классы
        class_ids = np.argmax(scores, axis=1)
        
        # Конвертируем из YOLO формата в обычные координаты
        x_center, y_center, width, height = boxes.T
        
        x1 = (x_center - width / 2) / scale_factor
        y1 = (y_center - height / 2) / scale_factor
        x2 = (x_center + width / 2) / scale_factor
        y2 = (y_center + height / 2) / scale_factor
        
        # Ограничиваем координаты размерами изображения
        h, w = original_shape[:2]
        x1 = np.clip(x1, 0, w)
        y1 = np.clip(y1, 0, h)
        x2 = np.clip(x2, 0, w)
        y2 = np.clip(y2, 0, h)
        
        boxes = np.column_stack([x1, y1, x2, y2])
        
        return boxes, max_scores, class_ids
    
    def draw_detections(self, image, boxes, scores, class_ids):
        """Отрисовывает детекции на изображении"""
        img_with_detections = image.copy()
        
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            
            # Получаем цвет и название класса
            color = self.class_colors[class_id % len(self.class_colors)]
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}'
            
            # Рисуем прямоугольник
            cv2.rectangle(img_with_detections, (x1, y1), (x2, y2), color, 2)
            
            # Подготавливаем текст
            label = f'{class_name}: {score:.2f}'
            
            # Получаем размер текста
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Рисуем фон для текста
            cv2.rectangle(
                img_with_detections,
                (x1, y1 - text_height - baseline),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Рисуем текст
            cv2.putText(
                img_with_detections,
                label,
                (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return img_with_detections
    
    def predict(self, image_path, output_path=None):
        """
        Предсказание на одном изображении
        
        Args:
            image_path: путь к изображению
            output_path: путь для сохранения результата
            
        Returns:
            boxes, scores, class_ids: результаты детекции
        """
        # Предобработка
        input_image, original_image, scale = self.preprocess_image(image_path)
        
        # Инференс
        outputs = self.session.run(self.output_names, {self.input_name: input_image})
        
        # Постобработка
        boxes, scores, class_ids = self.postprocess_outputs(
            outputs, scale, original_image.shape
        )
        
        print(f"🔍 Найдено детекций: {len(boxes)}")
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}'
            print(f"   {i+1}. {class_name}: {score:.3f}")
        
        # Отрисовываем детекции
        if len(boxes) > 0:
            img_with_detections = self.draw_detections(original_image, boxes, scores, class_ids)
            
            if output_path:
                cv2.imwrite(output_path, img_with_detections)
                print(f"✅ Результат сохранен: {output_path}")
        
        return boxes, scores, class_ids

def main():
    """Главная функция"""
    print("=" * 60)
    print("🚀 ИНФЕРЕНС С ONNX МОДЕЛЬЮ YOLOv8")
    print("=" * 60)
    
    # Путь к ONNX модели
    onnx_path = "exports/best.onnx"
    
    # Проверяем существование модели
    if not os.path.exists(onnx_path):
        print(f"❌ ONNX модель не найдена: {onnx_path}")
        print("   Сначала запустите export_onnx.py для экспорта модели")
        return
    
    # Создаем инференс объект
    try:
        inference = YOLOv8ONNXInference(onnx_path, conf_threshold=0.5)
    except Exception as e:
        print(f"❌ Ошибка при загрузке ONNX модели: {e}")
        return
    
    # Тестовое изображение
    test_image = "test_images/car1.jpg"
    
    if os.path.exists(test_image):
        print(f"\n🎯 Тестируем на изображении: {test_image}")
        
        # Создаем выходную директорию
        os.makedirs("onnx_results", exist_ok=True)
        output_path = "onnx_results/onnx_detection.jpg"
        
        # Выполняем предсказание
        boxes, scores, class_ids = inference.predict(test_image, output_path)
        
        print(f"\n✅ Инференс завершен!")
        print(f"   Результат сохранен: {output_path}")
    else:
        print(f"❌ Тестовое изображение не найдено: {test_image}")
        print("   Создайте директорию test_images/ и поместите туда car1.jpg")

if __name__ == "__main__":
    main()
