#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для скачивания и подготовки датасета с Kaggle
Автор: AI Assistant
"""

import os
import shutil
import yaml
from pathlib import Path
import random
import kagglehub

def download_car_damage_dataset():
    """Скачивает датасет car-damage-detection с Kaggle"""
    print("📥 Скачиваем датасет car-damage-detection с Kaggle...")
    
    try:
        # Скачиваем датасет
        path = kagglehub.dataset_download("anujms/car-damage-detection")
        print(f"✅ Датасет скачан в: {path}")
        return path
    except Exception as e:
        print(f"❌ Ошибка при скачивании: {e}")
        print("   Убедитесь, что установлен kagglehub: pip install kagglehub")
        return None

def prepare_dataset_structure():
    """Создает структуру директорий для YOLO датасета"""
    print("📁 Создаем структуру датасета...")
    
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
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}/")

def convert_dataset_to_yolo(kaggle_path, output_path="dataset"):
    """Конвертирует скачанный датасет в формат YOLO"""
    print(f"🔄 Конвертируем датасет в формат YOLO...")
    
    # Создаем структуру
    prepare_dataset_structure()
    
    # Маппинг классов (адаптируем под наш формат)
    class_mapping = {
        'dent': 2,      # dented
        'scratch': 3,   # scratched  
        'dirty': 1,     # dirty
        'clean': 0,     # clean
        'broken': 4,    # broken
        'damage': 4,    # broken (альтернативное название)
    }
    
    # Находим все изображения
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(kaggle_path).glob(f"**/*{ext}"))
        image_files.extend(Path(kaggle_path).glob(f"**/*{ext.upper()}"))
    
    print(f"   Найдено изображений: {len(image_files)}")
    
    if len(image_files) == 0:
        print("❌ Изображения не найдены в скачанном датасете!")
        return False
    
    # Разделяем на train/val/test (80/10/10)
    random.shuffle(image_files)
    train_split = int(0.8 * len(image_files))
    val_split = int(0.9 * len(image_files))
    
    train_files = image_files[:train_split]
    val_files = image_files[train_split:val_split]
    test_files = image_files[val_split:]
    
    print(f"   Распределение: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")
    
    # Обрабатываем файлы
    for i, files in enumerate([train_files, val_files, test_files]):
        split_name = ['train', 'val', 'test'][i]
        
        for img_file in files:
            try:
                # Копируем изображение
                dest_img = f"{output_path}/images/{split_name}/{img_file.name}"
                shutil.copy2(img_file, dest_img)
                
                # Ищем файл аннотаций
                label_file = img_file.with_suffix('.txt')
                if not label_file.exists():
                    # Пробуем найти в подпапках
                    for subdir in ['labels', 'annotations', 'yolo']:
                        potential_label = img_file.parent / subdir / f"{img_file.stem}.txt"
                        if potential_label.exists():
                            label_file = potential_label
                            break
                
                # Создаем аннотацию
                dest_label = f"{output_path}/labels/{split_name}/{img_file.stem}.txt"
                if label_file.exists():
                    convert_annotation_file(label_file, dest_label, class_mapping)
                else:
                    # Создаем пустую аннотацию (для тестирования)
                    with open(dest_label, "w") as f:
                        f.write("")
                        
            except Exception as e:
                print(f"⚠️  Ошибка при обработке {img_file}: {e}")
    
    print(f"✅ Датасет конвертирован в {output_path}/")
    return True

def convert_annotation_file(input_file, output_file, class_mapping):
    """Конвертирует файл аннотаций в формат YOLO"""
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        converted_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) >= 5:
                # Проверяем, является ли первый элемент числом (уже YOLO формат)
                try:
                    int(parts[0])
                    # Уже в YOLO формате
                    converted_lines.append(line)
                except ValueError:
                    # Нужна конвертация
                    class_name = parts[0].lower()
                    if class_name in class_mapping:
                        class_id = class_mapping[class_name]
                        converted_line = f"{class_id} {' '.join(parts[1:])}"
                        converted_lines.append(converted_line)
            else:
                # Пробуем найти класс по ключевым словам
                line_lower = line.lower()
                for class_name, class_id in class_mapping.items():
                    if class_name in line_lower:
                        # Создаем простую аннотацию
                        converted_line = f"{class_id} 0.5 0.5 0.1 0.1"
                        converted_lines.append(converted_line)
                        break
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(converted_lines))
            
    except Exception as e:
        print(f"⚠️  Ошибка при конвертации {input_file}: {e}")
        # Создаем пустой файл
        with open(output_file, 'w') as f:
            f.write("")

def create_sample_annotations():
    """Создает примеры аннотаций для тестирования"""
    print("📝 Создаем примеры аннотаций...")
    
    # Создаем несколько тестовых изображений с аннотациями
    from create_test_image import create_test_car_image, create_clean_car_image
    
    # Тестовые изображения
    create_test_car_image("dataset/images/train/car_damaged_1.jpg")
    create_clean_car_image("dataset/images/train/car_clean_1.jpg")
    create_test_car_image("dataset/images/val/car_damaged_val.jpg")
    create_clean_car_image("dataset/images/test/car_clean_test.jpg")
    
    # Аннотации для поврежденного автомобиля
    damaged_annotations = [
        "1 0.3 0.4 0.1 0.1",  # dirty
        "3 0.2 0.3 0.15 0.05",  # scratch
        "2 0.1 0.5 0.08 0.12",  # dent
        "4 0.4 0.2 0.1 0.08",  # broken
    ]
    
    # Аннотации для чистого автомобиля
    clean_annotations = [
        "0 0.5 0.5 0.8 0.6",  # clean
    ]
    
    # Сохраняем аннотации
    with open("dataset/labels/train/car_damaged_1.txt", "w") as f:
        f.write("\n".join(damaged_annotations))
    
    with open("dataset/labels/train/car_clean_1.txt", "w") as f:
        f.write(clean_annotations[0])
    
    with open("dataset/labels/val/car_damaged_val.txt", "w") as f:
        f.write("\n".join(damaged_annotations))
    
    with open("dataset/labels/test/car_clean_test.txt", "w") as f:
        f.write(clean_annotations[0])

def main():
    """Главная функция"""
    print("=" * 60)
    print("📥 СКАЧИВАНИЕ И ПОДГОТОВКА ДАТАСЕТА С KAGGLE")
    print("=" * 60)
    
    print("\nВыберите вариант:")
    print("1. Скачать датасет с Kaggle и конвертировать")
    print("2. Создать только пример датасета (быстро)")
    print("3. Показать инструкции")
    
    choice = input("\nВведите номер (1-3): ").strip()
    
    if choice == "1":
        # Скачиваем датасет
        kaggle_path = download_car_damage_dataset()
        if kaggle_path:
            # Конвертируем
            if convert_dataset_to_yolo(kaggle_path):
                print("\n🎉 Датасет готов! Теперь можно запустить: python train_yolo.py")
            else:
                print("\n⚠️  Ошибка при конвертации. Создаем пример датасета...")
                create_sample_annotations()
                print("✅ Пример датасета создан!")
    
    elif choice == "2":
        # Создаем пример датасета
        prepare_dataset_structure()
        create_sample_annotations()
        print("✅ Пример датасета создан!")
        print("   Теперь можно запустить: python train_yolo.py")
    
    elif choice == "3":
        show_instructions()
    
    else:
        print("❌ Неверный выбор!")

def show_instructions():
    """Показывает инструкции"""
    print("\n" + "="*60)
    print("📋 ИНСТРУКЦИИ")
    print("="*60)
    
    print("\n🔧 Установка kagglehub:")
    print("pip install kagglehub")
    
    print("\n📥 Скачивание датасета:")
    print("python download_kaggle_dataset.py")
    print("Выберите вариант 1")
    
    print("\n🚀 Обучение модели:")
    print("python train_yolo.py")
    
    print("\n🔍 Тестирование:")
    print("python detect.py")

if __name__ == "__main__":
    main()
