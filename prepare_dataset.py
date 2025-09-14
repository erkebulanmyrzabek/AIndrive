#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для подготовки датасета из различных источников
Автор: AI Assistant
"""

import os
import shutil
import yaml
from pathlib import Path
import random

def create_sample_dataset():
    """Создает пример датасета для демонстрации"""
    print("🎨 Создаем пример датасета...")
    
    # Создаем тестовые изображения
    from create_test_image import create_test_car_image, create_clean_car_image
    
    # Создаем несколько вариантов
    create_test_car_image("dataset/images/train/car_damaged_1.jpg")
    create_clean_car_image("dataset/images/train/car_clean_1.jpg")
    create_test_car_image("dataset/images/train/car_damaged_2.jpg")
    create_clean_car_image("dataset/images/train/car_clean_2.jpg")
    
    # Создаем валидационные изображения
    create_test_car_image("dataset/images/val/car_damaged_val_1.jpg")
    create_clean_car_image("dataset/images/val/car_clean_val_1.jpg")
    
    # Создаем тестовые изображения
    create_test_car_image("dataset/images/test/car_damaged_test_1.jpg")
    create_clean_car_image("dataset/images/test/car_clean_test_1.jpg")
    
    # Создаем аннотации (примеры)
    create_sample_annotations()
    
    print("✅ Пример датасета создан!")

def create_sample_annotations():
    """Создает примеры аннотаций для тестовых изображений"""
    
    # Аннотации для поврежденного автомобиля
    damaged_annotations = [
        "1 0.3 0.4 0.1 0.1",  # dirty spot
        "3 0.2 0.3 0.15 0.05",  # scratch
        "2 0.1 0.5 0.08 0.12",  # dent
        "4 0.4 0.2 0.1 0.08",  # broken glass
    ]
    
    # Аннотации для чистого автомобиля
    clean_annotations = [
        "0 0.5 0.5 0.8 0.6",  # clean car
    ]
    
    # Сохраняем аннотации
    with open("dataset/labels/train/car_damaged_1.txt", "w") as f:
        f.write("\n".join(damaged_annotations))
    
    with open("dataset/labels/train/car_clean_1.txt", "w") as f:
        f.write(clean_annotations[0])
    
    with open("dataset/labels/train/car_damaged_2.txt", "w") as f:
        f.write("\n".join(damaged_annotations))
    
    with open("dataset/labels/train/car_clean_2.txt", "w") as f:
        f.write(clean_annotations[0])
    
    # Валидационные аннотации
    with open("dataset/labels/val/car_damaged_val_1.txt", "w") as f:
        f.write("\n".join(damaged_annotations))
    
    with open("dataset/labels/val/car_clean_val_1.txt", "w") as f:
        f.write(clean_annotations[0])
    
    # Тестовые аннотации
    with open("dataset/labels/test/car_damaged_test_1.txt", "w") as f:
        f.write("\n".join(damaged_annotations))
    
    with open("dataset/labels/test/car_clean_test_1.txt", "w") as f:
        f.write(clean_annotations[0])

def convert_kaggle_dataset(kaggle_path, output_path="dataset"):
    """Конвертирует датасет с Kaggle в формат YOLO"""
    print(f"🔄 Конвертируем датасет из {kaggle_path}...")
    
    # Создаем структуру
    os.makedirs(f"{output_path}/images/train", exist_ok=True)
    os.makedirs(f"{output_path}/images/val", exist_ok=True)
    os.makedirs(f"{output_path}/images/test", exist_ok=True)
    os.makedirs(f"{output_path}/labels/train", exist_ok=True)
    os.makedirs(f"{output_path}/labels/val", exist_ok=True)
    os.makedirs(f"{output_path}/labels/test", exist_ok=True)
    
    # Маппинг классов (адаптируйте под ваш датасет)
    class_mapping = {
        'dent': 2,      # dented
        'scratch': 3,   # scratched
        'dirty': 1,     # dirty
        'clean': 0,     # clean
        'broken': 4,    # broken
    }
    
    # Обрабатываем изображения
    image_files = list(Path(kaggle_path).glob("**/*.jpg")) + list(Path(kaggle_path).glob("**/*.png"))
    
    # Разделяем на train/val/test (80/10/10)
    random.shuffle(image_files)
    train_split = int(0.8 * len(image_files))
    val_split = int(0.9 * len(image_files))
    
    train_files = image_files[:train_split]
    val_files = image_files[train_split:val_split]
    test_files = image_files[val_split:]
    
    # Копируем файлы
    for i, files in enumerate([train_files, val_files, test_files]):
        split_name = ['train', 'val', 'test'][i]
        
        for img_file in files:
            # Копируем изображение
            shutil.copy2(img_file, f"{output_path}/images/{split_name}/")
            
            # Ищем соответствующий файл аннотаций
            label_file = img_file.with_suffix('.txt')
            if label_file.exists():
                # Конвертируем аннотации
                convert_annotation_file(label_file, f"{output_path}/labels/{split_name}/{img_file.name}.txt", class_mapping)
            else:
                # Создаем пустую аннотацию
                with open(f"{output_path}/labels/{split_name}/{img_file.name}.txt", "w") as f:
                    f.write("")
    
    print(f"✅ Датасет конвертирован в {output_path}/")

def convert_annotation_file(input_file, output_file, class_mapping):
    """Конвертирует файл аннотаций в формат YOLO"""
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        converted_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_name = parts[0]
                if class_name in class_mapping:
                    class_id = class_mapping[class_name]
                    # Остальные координаты оставляем как есть
                    converted_line = f"{class_id} {' '.join(parts[1:])}"
                    converted_lines.append(converted_line)
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(converted_lines))
            
    except Exception as e:
        print(f"⚠️  Ошибка при конвертации {input_file}: {e}")
        # Создаем пустой файл
        with open(output_file, 'w') as f:
            f.write("")

def download_sample_dataset():
    """Скачивает небольшой пример датасета"""
    print("📥 Скачиваем пример датасета...")
    
    # Создаем пример датасета
    create_sample_dataset()
    
    print("✅ Пример датасета готов!")
    print("   Теперь можно запустить train_yolo.py")

def main():
    """Главная функция"""
    print("=" * 60)
    print("📊 ПОДГОТОВКА ДАТАСЕТА ДЛЯ YOLOv8")
    print("=" * 60)
    
    print("\nВыберите вариант:")
    print("1. Создать пример датасета (для тестирования)")
    print("2. Конвертировать датасет с Kaggle")
    print("3. Показать инструкции по скачиванию")
    
    choice = input("\nВведите номер (1-3): ").strip()
    
    if choice == "1":
        download_sample_dataset()
    elif choice == "2":
        kaggle_path = input("Введите путь к скачанному датасету: ").strip()
        if os.path.exists(kaggle_path):
            convert_kaggle_dataset(kaggle_path)
        else:
            print("❌ Путь не найден!")
    elif choice == "3":
        show_download_instructions()
    else:
        print("❌ Неверный выбор!")

def show_download_instructions():
    """Показывает инструкции по скачиванию датасета"""
    print("\n" + "="*60)
    print("📥 ИНСТРУКЦИИ ПО СКАЧИВАНИЮ ДАТАСЕТА")
    print("="*60)
    
    print("\n🔗 Рекомендуемые датасеты:")
    print("1. Car Damage Detection: https://www.kaggle.com/datasets/sshikamaru/car-damage-detection")
    print("2. Vehicle Damage Dataset: https://www.kaggle.com/datasets/ravirajsinh45/real-time-car-detection")
    
    print("\n📋 Пошаговая инструкция:")
    print("1. Зарегистрируйтесь на Kaggle.com")
    print("2. Перейдите на страницу датасета")
    print("3. Нажмите 'Download' (требуется регистрация)")
    print("4. Распакуйте архив в папку 'downloaded_dataset/'")
    print("5. Запустите: python prepare_dataset.py")
    print("6. Выберите вариант 2 и укажите путь к датасету")
    
    print("\n⚡ Быстрый старт:")
    print("1. Запустите: python prepare_dataset.py")
    print("2. Выберите вариант 1 (создать пример)")
    print("3. Запустите: python train_yolo.py")

if __name__ == "__main__":
    main()
