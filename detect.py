#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для детекции состояния автомобиля с помощью обученной модели YOLOv8
Автор: AI Assistant
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path

# Цвета для каждого класса (BGR формат для OpenCV)
CLASS_COLORS = {
    0: (0, 255, 0),      # clean - зеленый
    1: (0, 165, 255),    # dirty - оранжевый  
    2: (0, 0, 255),      # dented - красный
    3: (255, 0, 0),      # scratched - синий
    4: (128, 0, 128),    # broken - фиолетовый
}

# Названия классов
CLASS_NAMES = {
    0: 'clean',
    1: 'dirty', 
    2: 'dented',
    3: 'scratched',
    4: 'broken'
}

def load_model(model_path):
    """Загружает обученную модель YOLOv8"""
    try:
        if not os.path.exists(model_path):
            print(f"❌ Модель не найдена: {model_path}")
            print("   Сначала запустите train_yolo.py для обучения модели")
            return None
        
        print(f"📥 Загружаем модель: {model_path}")
        model = YOLO(model_path)
        print("✅ Модель загружена успешно")
        return model
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели: {e}")
        return None

def draw_detections(image, boxes, scores, class_ids, class_names):
    """Отрисовывает детекции на изображении"""
    img_with_detections = image.copy()
    
    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        # Получаем координаты bbox
        x1, y1, x2, y2 = map(int, box)
        
        # Получаем цвет и название класса
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        class_name = class_names.get(class_id, f'class_{class_id}')
        
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

def detect_single_image(model, image_path, output_dir, conf_threshold=0.5):
    """Детектирует состояние автомобиля на одном изображении"""
    print(f"🔍 Обрабатываем изображение: {image_path}")
    
    # Проверяем существование файла
    if not os.path.exists(image_path):
        print(f"❌ Изображение не найдено: {image_path}")
        return False
    
    try:
        # Загружаем изображение
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Не удалось загрузить изображение: {image_path}")
            return False
        
        print(f"   Размер изображения: {image.shape[1]}x{image.shape[0]}")
        
        # Выполняем детекцию
        results = model(image, conf=conf_threshold, verbose=False)
        
        # Обрабатываем результаты
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # Координаты bbox
            scores = results[0].boxes.conf.cpu().numpy()  # Уверенность
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # ID классов
            
            print(f"   Найдено детекций: {len(boxes)}")
            
            # Выводим информацию о детекциях
            for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                class_name = CLASS_NAMES.get(class_id, f'class_{class_id}')
                print(f"     {i+1}. {class_name}: {score:.3f}")
            
            # Отрисовываем детекции
            img_with_detections = draw_detections(image, boxes, scores, class_ids, CLASS_NAMES)
        else:
            print("   Детекции не найдены")
            img_with_detections = image.copy()
        
        # Создаем выходную директорию
        os.makedirs(output_dir, exist_ok=True)
        
        # Сохраняем результат
        output_path = os.path.join(output_dir, f"detected_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, img_with_detections)
        print(f"   Результат сохранен: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при обработке изображения: {e}")
        return False

def detect_batch(model, input_dir, output_dir, conf_threshold=0.5):
    """Детектирует состояние автомобиля на всех изображениях в директории"""
    print(f"📁 Обрабатываем все изображения в: {input_dir}")
    
    if not os.path.exists(input_dir):
        print(f"❌ Директория не найдена: {input_dir}")
        return False
    
    # Поддерживаемые форматы изображений
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Находим все изображения
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"❌ Изображения не найдены в: {input_dir}")
        return False
    
    print(f"   Найдено изображений: {len(image_files)}")
    
    # Обрабатываем каждое изображение
    success_count = 0
    for image_path in image_files:
        if detect_single_image(model, str(image_path), output_dir, conf_threshold):
            success_count += 1
    
    print(f"✅ Успешно обработано: {success_count}/{len(image_files)} изображений")
    return success_count > 0

def main():
    """Главная функция"""
    print("=" * 60)
    print("🚗 ДЕТЕКЦИЯ СОСТОЯНИЯ АВТОМОБИЛЯ С YOLOv8")
    print("=" * 60)
    
    # Путь к обученной модели
    model_path = "runs/damage/weights/best.pt"
    
    # Загружаем модель
    model = load_model(model_path)
    if model is None:
        return
    
    # Создаем выходную директорию
    output_dir = "runs/damage_predict"
    os.makedirs(output_dir, exist_ok=True)
    
    # Проверяем наличие тестового изображения
    test_image = "test_images/car1.jpg"
    
    if os.path.exists(test_image):
        print(f"\n🎯 Тестируем на изображении: {test_image}")
        success = detect_single_image(model, test_image, output_dir, conf_threshold=0.5)
        
        if success:
            print("✅ Детекция выполнена успешно!")
        else:
            print("❌ Ошибка при детекции!")
    else:
        print(f"\n⚠️  Тестовое изображение не найдено: {test_image}")
        print("   Создайте директорию test_images/ и поместите туда car1.jpg")
        
        # Пробуем найти другие изображения
        test_dir = "test_images"
        if os.path.exists(test_dir):
            print(f"   Обрабатываем все изображения в {test_dir}/")
            detect_batch(model, test_dir, output_dir, conf_threshold=0.5)
    
    print(f"\n📁 Результаты сохранены в: {output_dir}/")
    print("   Для просмотра результатов откройте изображения в этой папке")

if __name__ == "__main__":
    main()
