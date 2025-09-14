#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обучение модели для inDrive кейса: определение состояния автомобиля
Автор: AI Assistant
"""

import os
import yaml
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
from datetime import datetime

# Классы для inDrive кейса
INDRIVE_CLASSES = {
    0: {"name": "clean_intact", "description": "Чистый и целый", "priority": "high"},
    1: {"name": "clean_damaged", "description": "Чистый, но поврежденный", "priority": "medium"},
    2: {"name": "dirty_intact", "description": "Грязный, но целый", "priority": "medium"},
    3: {"name": "dirty_damaged", "description": "Грязный и поврежденный", "priority": "low"},
    4: {"name": "very_dirty", "description": "Очень грязный", "priority": "low"},
    5: {"name": "severely_damaged", "description": "Сильно поврежденный", "priority": "critical"}
}

def check_gpu():
    """Проверяет доступность GPU"""
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
        with open('indrive_data.yaml', 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        print("✅ Конфигурация датасета загружена успешно")
        print(f"   Классы: {data_config['names']}")
        return data_config
    except FileNotFoundError:
        print("❌ Файл indrive_data.yaml не найден!")
        return None
    except Exception as e:
        print(f"❌ Ошибка при загрузке indrive_data.yaml: {e}")
        return None

def create_indrive_dataset_structure():
    """Создает структуру датасета для inDrive"""
    print("📁 Создаем структуру датасета для inDrive...")
    
    directories = [
        "indrive_dataset",
        "indrive_dataset/images",
        "indrive_dataset/images/train",
        "indrive_dataset/images/val", 
        "indrive_dataset/images/test",
        "indrive_dataset/labels",
        "indrive_dataset/labels/train",
        "indrive_dataset/labels/val",
        "indrive_dataset/labels/test",
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ {directory}/")
    
    # Создаем README для датасета
    readme_content = """# inDrive Dataset - Определение состояния автомобиля

## Описание классов

### Основные категории:
- **Чистота**: clean (чистый) vs dirty (грязный) vs very_dirty (очень грязный)
- **Целостность**: intact (целый) vs damaged (поврежденный) vs severely_damaged (сильно поврежденный)

### Комбинированные классы:
0. **clean_intact** - Чистый и целый (идеальное состояние)
1. **clean_damaged** - Чистый, но поврежденный (требует внимания)
2. **dirty_intact** - Грязный, но целый (нужна мойка)
3. **dirty_damaged** - Грязный и поврежденный (мойка + ремонт)
4. **very_dirty** - Очень грязный (критическое состояние)
5. **severely_damaged** - Сильно поврежденный (небезопасен)

## Формат аннотаций YOLO:
```
class_id center_x center_y width height
```

## Приоритеты для inDrive:
- **high**: clean_intact (можно принимать заказы)
- **medium**: clean_damaged, dirty_intact (требует внимания)
- **low**: dirty_damaged, very_dirty (не рекомендуется)
- **critical**: severely_damaged (заблокировать)
"""
    
    with open("indrive_dataset/README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("   ✅ indrive_dataset/README.md")

def train_indrive_model():
    """Обучает модель для inDrive кейса"""
    print("🚀 Начинаем обучение модели для inDrive...")
    
    # Проверяем GPU
    use_gpu = check_gpu()
    
    # Загружаем конфигурацию
    data_config = load_data_config()
    if not data_config:
        return False
    
    # Создаем структуру датасета
    create_indrive_dataset_structure()
    
    # Проверяем существование датасета
    dataset_path = data_config.get('path', './indrive_dataset')
    if not os.path.exists(dataset_path):
        print(f"❌ Директория датасета не найдена: {dataset_path}")
        print("   Создайте структуру датасета согласно indrive_data.yaml")
        return False
    
    try:
        # Загружаем предобученную модель
        print("📥 Загружаем предобученную модель YOLOv8n...")
        model = YOLO('yolov8n.pt')
        
        # Параметры обучения для inDrive
        training_args = {
            'data': 'indrive_data.yaml',
            'epochs': 100,  # Больше эпох для лучшего качества
            'imgsz': 640,
            'batch': 16 if use_gpu else 8,
            'device': 'cuda' if use_gpu else 'cpu',
            'project': 'indrive_runs',
            'name': 'car_condition',
            'save': True,
            'save_period': 20,
            'cache': True,
            'workers': 8 if use_gpu else 4,
            'patience': 15,  # Больше терпения для сложной задачи
            'lr0': 0.01,
            'lrf': 0.1,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 5,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'val': True,
            'plots': True,
            'verbose': True,
            # Дополнительные параметры для inDrive
            'augment': True,  # Аугментация для разнообразия
            'mixup': 0.1,     # Mixup для лучшей генерализации
            'copy_paste': 0.1, # Copy-paste аугментация
            'degrees': 10,    # Поворот изображений
            'translate': 0.1, # Сдвиг
            'scale': 0.5,     # Масштабирование
            'shear': 2.0,     # Наклон
            'perspective': 0.0, # Перспектива
            'flipud': 0.0,    # Вертикальное отражение
            'fliplr': 0.5,    # Горизонтальное отражение
            'mosaic': 1.0,    # Мозаика
            'mixup': 0.1,     # Mixup
        }
        
        print("🎯 Параметры обучения для inDrive:")
        for key, value in training_args.items():
            print(f"   {key}: {value}")
        
        # Начинаем обучение
        print("\n🔥 Запускаем обучение...")
        results = model.train(**training_args)
        
        print("✅ Обучение завершено успешно!")
        print(f"   Результаты сохранены в: indrive_runs/car_condition/")
        print(f"   Лучшая модель: indrive_runs/car_condition/weights/best.pt")
        
        # Анализируем результаты
        analyze_training_results(results, 'indrive_runs/car_condition/')
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при обучении: {e}")
        return False

def analyze_training_results(results, output_dir):
    """Анализирует результаты обучения"""
    print("\n📊 Анализ результатов обучения...")
    
    try:
        # Создаем директорию для анализа
        analysis_dir = os.path.join(output_dir, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Загружаем метрики
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            
            # Создаем отчет
            report = {
                'timestamp': datetime.now().isoformat(),
                'model': 'YOLOv8n for inDrive',
                'classes': INDRIVE_CLASSES,
                'metrics': {
                    'mAP50': metrics.get('metrics/mAP50(B)', 0),
                    'mAP50-95': metrics.get('metrics/mAP50-95(B)', 0),
                    'precision': metrics.get('metrics/precision(B)', 0),
                    'recall': metrics.get('metrics/recall(B)', 0),
                    'f1': metrics.get('metrics/f1(B)', 0)
                }
            }
            
            # Сохраняем отчет
            with open(os.path.join(analysis_dir, 'training_report.json'), 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            print("📈 Финальные метрики:")
            print(f"   mAP50: {report['metrics']['mAP50']:.3f}")
            print(f"   mAP50-95: {report['metrics']['mAP50-95']:.3f}")
            print(f"   Precision: {report['metrics']['precision']:.3f}")
            print(f"   Recall: {report['metrics']['recall']:.3f}")
            print(f"   F1: {report['metrics']['f1']:.3f}")
            
            # Оценка качества для inDrive
            evaluate_indrive_quality(report['metrics'])
        
    except Exception as e:
        print(f"⚠️  Ошибка при анализе результатов: {e}")

def evaluate_indrive_quality(metrics):
    """Оценивает качество модели для inDrive"""
    print("\n🎯 Оценка качества для inDrive:")
    
    mAP50 = metrics.get('mAP50', 0)
    precision = metrics.get('precision', 0)
    recall = metrics.get('recall', 0)
    
    # Критерии для inDrive
    if mAP50 >= 0.8 and precision >= 0.8 and recall >= 0.8:
        print("   ✅ Отличное качество - готово для production")
        print("   💡 Рекомендация: Можно интегрировать в приложение inDrive")
    elif mAP50 >= 0.7 and precision >= 0.7 and recall >= 0.7:
        print("   ⚠️  Хорошее качество - требует доработки")
        print("   💡 Рекомендация: Собрать больше данных, особенно для критических классов")
    else:
        print("   ❌ Низкое качество - требует значительной доработки")
        print("   💡 Рекомендация: Пересмотреть подход, увеличить датасет")
    
    # Анализ по классам
    print("\n📋 Анализ по классам для inDrive:")
    for class_id, info in INDRIVE_CLASSES.items():
        priority = info['priority']
        if priority == 'critical':
            print(f"   🚨 {info['name']}: {info['description']} - КРИТИЧЕСКИЙ класс")
        elif priority == 'high':
            print(f"   ✅ {info['name']}: {info['description']} - ВЫСОКИЙ приоритет")
        elif priority == 'medium':
            print(f"   ⚠️  {info['name']}: {info['description']} - СРЕДНИЙ приоритет")
        else:
            print(f"   📉 {info['name']}: {info['description']} - НИЗКИЙ приоритет")

def create_indrive_demo_data():
    """Создает демонстрационные данные для inDrive"""
    print("🎨 Создаем демонстрационные данные для inDrive...")
    
    from create_test_image import create_test_car_image, create_clean_car_image
    
    # Создаем примеры для каждого класса
    demo_images = [
        ("clean_intact", "Чистый и целый автомобиль"),
        ("clean_damaged", "Чистый, но с царапинами"),
        ("dirty_intact", "Грязный, но целый"),
        ("dirty_damaged", "Грязный и поврежденный"),
        ("very_dirty", "Очень грязный"),
        ("severely_damaged", "Сильно поврежденный")
    ]
    
    for i, (class_name, description) in enumerate(demo_images):
        # Создаем изображение
        if "clean" in class_name and "intact" in class_name:
            create_clean_car_image(f"indrive_dataset/images/train/demo_{class_name}.jpg")
        else:
            create_test_car_image(f"indrive_dataset/images/train/demo_{class_name}.jpg")
        
        # Создаем аннотацию
        with open(f"indrive_dataset/labels/train/demo_{class_name}.txt", "w") as f:
            f.write(f"{i} 0.5 0.5 0.8 0.6")  # Центральный bbox
        
        print(f"   ✅ Создан демо: {class_name} - {description}")

def main():
    """Главная функция"""
    print("=" * 70)
    print("🚗 ОБУЧЕНИЕ МОДЕЛИ ДЛЯ inDrive КЕЙСА")
    print("   Определение состояния автомобиля: чистота + целостность")
    print("=" * 70)
    
    # Проверяем наличие необходимых файлов
    if not os.path.exists('indrive_data.yaml'):
        print("❌ Файл indrive_data.yaml не найден!")
        return
    
    # Создаем демо данные
    create_indrive_demo_data()
    
    # Запускаем обучение
    success = train_indrive_model()
    
    if success:
        print("\n🎉 Обучение завершено успешно!")
        print("   Теперь можно использовать app_indrive.py для демонстрации")
        print("   Запустите: python app_indrive.py")
    else:
        print("\n💥 Обучение завершилось с ошибкой!")

if __name__ == "__main__":
    main()
