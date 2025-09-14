#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для тестирования обученной модели на различных изображениях
Автор: AI Assistant
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path

def test_model_performance(model_path, test_dir="test_images", output_dir="test_results"):
    """Тестирует производительность модели на тестовых изображениях"""
    print(f"🧪 Тестируем модель: {model_path}")
    
    # Загружаем модель
    model = YOLO(model_path)
    
    # Создаем выходную директорию
    os.makedirs(output_dir, exist_ok=True)
    
    # Находим все изображения
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(test_dir).glob(f'*{ext}'))
        image_files.extend(Path(test_dir).glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"❌ Изображения не найдены в {test_dir}")
        return
    
    print(f"📊 Найдено изображений для тестирования: {len(image_files)}")
    
    # Классы и их цвета
    class_names = ['clean', 'dirty', 'dented', 'scratched', 'broken']
    class_colors = [(0, 255, 0), (0, 165, 255), (0, 0, 255), (255, 0, 0), (128, 0, 128)]
    
    # Статистика
    total_detections = 0
    class_counts = {i: 0 for i in range(5)}
    confidence_scores = []
    
    # Обрабатываем каждое изображение
    for i, image_path in enumerate(image_files):
        print(f"\n🔍 Обрабатываем: {image_path.name}")
        
        # Загружаем изображение
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"   ❌ Не удалось загрузить изображение")
            continue
        
        # Детекция
        results = model(image, conf=0.3, verbose=False)
        
        # Обрабатываем результаты
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            print(f"   Найдено детекций: {len(boxes)}")
            
            # Обновляем статистику
            total_detections += len(boxes)
            confidence_scores.extend(scores.tolist())
            
            for class_id in class_ids:
                class_counts[class_id] += 1
            
            # Рисуем детекции
            img_with_detections = image.copy()
            for box, score, class_id in zip(boxes, scores, class_ids):
                x1, y1, x2, y2 = map(int, box)
                color = class_colors[class_id]
                class_name = class_names[class_id]
                
                # Рисуем прямоугольник
                cv2.rectangle(img_with_detections, (x1, y1), (x2, y2), color, 2)
                
                # Рисуем подпись
                label = f'{class_name}: {score:.2f}'
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                cv2.rectangle(
                    img_with_detections,
                    (x1, y1 - text_height - baseline),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                
                cv2.putText(
                    img_with_detections,
                    label,
                    (x1, y1 - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
            
            # Сохраняем результат
            output_path = os.path.join(output_dir, f"test_{image_path.stem}.jpg")
            cv2.imwrite(output_path, img_with_detections)
            print(f"   ✅ Сохранено: {output_path}")
        else:
            print("   ℹ️  Детекции не найдены")
    
    # Выводим статистику
    print(f"\n📊 СТАТИСТИКА ТЕСТИРОВАНИЯ")
    print(f"=" * 40)
    print(f"Всего изображений: {len(image_files)}")
    print(f"Всего детекций: {total_detections}")
    print(f"Средняя уверенность: {np.mean(confidence_scores):.3f}")
    print(f"Медианная уверенность: {np.median(confidence_scores):.3f}")
    
    print(f"\nРаспределение по классам:")
    for class_id, count in class_counts.items():
        class_name = class_names[class_id]
        percentage = (count / total_detections * 100) if total_detections > 0 else 0
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    # Создаем график распределения уверенности
    if confidence_scores:
        plt.figure(figsize=(10, 6))
        plt.hist(confidence_scores, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Уверенность детекции')
        plt.ylabel('Количество')
        plt.title('Распределение уверенности детекций')
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(output_dir, 'confidence_distribution.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 График сохранен: {plot_path}")

def main():
    """Главная функция"""
    print("=" * 60)
    print("🧪 ТЕСТИРОВАНИЕ МОДЕЛИ YOLOv8")
    print("=" * 60)
    
    # Путь к модели
    model_path = "runs/damage/weights/best.pt"
    
    # Проверяем существование модели
    if not os.path.exists(model_path):
        print(f"❌ Модель не найдена: {model_path}")
        print("   Сначала запустите train_yolo.py для обучения модели")
        return
    
    # Проверяем наличие тестовых изображений
    test_dir = "test_images"
    if not os.path.exists(test_dir):
        print(f"❌ Директория с тестовыми изображениями не найдена: {test_dir}")
        print("   Создайте директорию test_images/ и поместите туда изображения")
        return
    
    # Запускаем тестирование
    test_model_performance(model_path, test_dir, "test_results")
    
    print(f"\n🎉 Тестирование завершено!")
    print(f"   Результаты сохранены в: test_results/")

if __name__ == "__main__":
    main()
