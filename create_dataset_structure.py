#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для создания структуры датасета YOLO
Автор: AI Assistant
"""

import os
from pathlib import Path

def create_dataset_structure():
    """Создает структуру директорий для датасета YOLO"""
    print("📁 Создаем структуру датасета...")
    
    # Основные директории
    directories = [
        "dataset",
        "dataset/images",
        "dataset/images/train",
        "dataset/images/val", 
        "dataset/images/test",
        "dataset/labels",
        "dataset/labels/train",
        "dataset/labels/val",
        "dataset/labels/test",
        "test_images",
        "runs",
        "exports"
    ]
    
    # Создаем директории
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}/")
    
    # Создаем README для датасета
    dataset_readme = """# Структура датасета

## Формат аннотаций YOLO

Каждое изображение должно иметь соответствующий .txt файл с аннотациями.

### Формат файла аннотаций:
```
class_id center_x center_y width height
```

Где:
- class_id: ID класса (0-4)
- center_x, center_y: координаты центра bbox (нормализованные 0-1)
- width, height: ширина и высота bbox (нормализованные 0-1)

### Классы:
- 0: clean (чистый)
- 1: dirty (грязный)
- 2: dented (с вмятинами)
- 3: scratched (с царапинами)
- 4: broken (битый)

### Пример аннотации:
```
2 0.5 0.5 0.3 0.4
```
Означает: класс 2 (dented), центр в (0.5, 0.5), размер 30%x40% от изображения

## Размещение файлов:
- Изображения: dataset/images/{train,val,test}/
- Аннотации: dataset/labels/{train,val,test}/
- Имена файлов должны совпадать (например: car1.jpg и car1.txt)
"""
    
    with open("dataset/README.md", "w", encoding="utf-8") as f:
        f.write(dataset_readme)
    
    print("   ✅ dataset/README.md")
    
    print("\n🎉 Структура датасета создана!")
    print("   Теперь добавьте изображения и аннотации в соответствующие папки")

if __name__ == "__main__":
    create_dataset_structure()
