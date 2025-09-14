#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Быстрый старт - создает тестовый датасет и запускает обучение
Автор: AI Assistant
"""

import os
import subprocess
import sys

def create_quick_dataset():
    """Создает быстрый тестовый датасет"""
    print("🚀 БЫСТРЫЙ СТАРТ - СОЗДАНИЕ ТЕСТОВОГО ДАТАСЕТА")
    print("=" * 60)
    
    # 1. Создаем структуру датасета
    print("📁 Создаем структуру датасета...")
    subprocess.run([sys.executable, "create_dataset_structure.py"], check=True)
    
    # 2. Создаем тестовые изображения
    print("🎨 Создаем тестовые изображения...")
    subprocess.run([sys.executable, "create_test_image.py"], check=True)
    
    # 3. Создаем пример датасета
    print("📊 Создаем пример датасета...")
    subprocess.run([sys.executable, "prepare_dataset.py"], input="1\n", text=True, check=True)
    
    print("✅ Тестовый датасет готов!")
    print("   Теперь можно запустить обучение: python train_yolo.py")

def main():
    """Главная функция"""
    try:
        create_quick_dataset()
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка: {e}")
    except KeyboardInterrupt:
        print("\n⏹️  Прервано пользователем")
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")

if __name__ == "__main__":
    main()
