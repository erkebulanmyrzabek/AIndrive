#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для обучения YOLOv8 на распознавание состояния автомобиля
Автор: AI Assistant
"""

import os
import yaml
from ultralytics import YOLO
import torch

def check_gpu():
    """Проверяет доступность GPU для обучения"""
    if torch.cuda.is_available():
        print(f"✅ GPU доступен: {torch.cuda.get_device_name(0)}")
        print(f"   Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return True
    else:
        print("⚠️  GPU недоступен, будет использоваться CPU")
        return False

def load_data_config():
    """Загружает конфигурацию датасета"""
    try:
        with open('data.yaml', 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        print("✅ Конфигурация датасета загружена успешно")
        print(f"   Классы: {data_config['names']}")
        return data_config
    except FileNotFoundError:
        print("❌ Файл data.yaml не найден!")
        return None
    except Exception as e:
        print(f"❌ Ошибка при загрузке data.yaml: {e}")
        return None

def train_model():
    """Обучает модель YOLOv8"""
    print("🚀 Начинаем обучение модели YOLOv8...")
    
    # Проверяем доступность GPU
    use_gpu = check_gpu()
    
    # Загружаем конфигурацию датасета
    data_config = load_data_config()
    if not data_config:
        return False
    
    # Проверяем существование датасета
    dataset_path = data_config.get('path', './dataset')
    if not os.path.exists(dataset_path):
        print(f"❌ Директория датасета не найдена: {dataset_path}")
        print("   Создайте структуру датасета согласно data.yaml")
        return False
    
    try:
        # Загружаем предобученную модель YOLOv8n
        print("📥 Загружаем предобученную модель YOLOv8n...")
        model = YOLO('yolov8n.pt')  # Автоматически скачается при первом запуске
        
        # Параметры обучения
        training_args = {
            'data': 'data.yaml',           # Путь к конфигурации датасета
            'epochs': 50,                  # Количество эпох
            'imgsz': 640,                  # Размер изображений
            'batch': 16 if use_gpu else 8, # Размер батча (меньше для CPU)
            'device': 'cuda' if use_gpu else 'cpu',  # Устройство для обучения
            'project': 'runs',             # Директория для сохранения результатов
            'name': 'damage',              # Название эксперимента
            'save': True,                  # Сохранять чекпоинты
            'save_period': 10,             # Сохранять каждые 10 эпох
            'cache': True,                 # Кэшировать изображения
            'workers': 8 if use_gpu else 4, # Количество воркеров
            'patience': 10,                # Ранняя остановка после 10 эпох без улучшения
            'lr0': 0.01,                   # Начальная скорость обучения
            'lrf': 0.1,                    # Финальная скорость обучения
            'momentum': 0.937,             # Момент
            'weight_decay': 0.0005,        # Весовой распад
            'warmup_epochs': 3,            # Эпохи разогрева
            'warmup_momentum': 0.8,        # Момент разогрева
            'warmup_bias_lr': 0.1,         # Скорость обучения bias при разогреве
            'box': 7.5,                    # Вес box loss
            'cls': 0.5,                    # Вес classification loss
            'dfl': 1.5,                    # Вес DFL loss
            'pose': 12.0,                  # Вес pose loss (не используется)
            'kobj': 2.0,                   # Вес keypoint obj loss (не используется)
            'label_smoothing': 0.0,        # Сглаживание меток
            'nbs': 64,                     # Nominal batch size
            'overlap_mask': True,          # Перекрытие масок при обучении
            'mask_ratio': 4,               # Соотношение масок
            'dropout': 0.0,                # Dropout (не используется в YOLOv8)
            'val': True,                   # Валидация во время обучения
            'plots': True,                 # Создавать графики
            'verbose': True,               # Подробный вывод
        }
        
        print("🎯 Параметры обучения:")
        for key, value in training_args.items():
            print(f"   {key}: {value}")
        
        # Начинаем обучение
        print("\n🔥 Запускаем обучение...")
        results = model.train(**training_args)
        
        print("✅ Обучение завершено успешно!")
        print(f"   Результаты сохранены в: runs/damage/")
        print(f"   Лучшая модель: runs/damage/weights/best.pt")
        print(f"   Последняя модель: runs/damage/weights/last.pt")
        
        # Выводим метрики
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print(f"\n📊 Финальные метрики:")
            print(f"   mAP50: {metrics.get('metrics/mAP50(B)', 0):.3f}")
            print(f"   mAP50-95: {metrics.get('metrics/mAP50-95(B)', 0):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при обучении: {e}")
        return False

def main():
    """Главная функция"""
    print("=" * 60)
    print("🚗 ОБУЧЕНИЕ YOLOv8 ДЛЯ РАСПОЗНАВАНИЯ СОСТОЯНИЯ АВТОМОБИЛЯ")
    print("=" * 60)
    
    # Проверяем наличие необходимых файлов
    if not os.path.exists('data.yaml'):
        print("❌ Файл data.yaml не найден!")
        return
    
    # Запускаем обучение
    success = train_model()
    
    if success:
        print("\n🎉 Обучение завершено успешно!")
        print("   Теперь можно использовать detect.py для тестирования модели")
    else:
        print("\n💥 Обучение завершилось с ошибкой!")

if __name__ == "__main__":
    main()
