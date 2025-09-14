#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для запуска Streamlit приложения
Автор: AI Assistant
"""

import subprocess
import sys
import os

def check_requirements():
    """Проверяет установленные зависимости"""
    try:
        import streamlit
        import ultralytics
        import cv2
        import plotly
        print("✅ Все зависимости установлены")
        return True
    except ImportError as e:
        print(f"❌ Отсутствует зависимость: {e}")
        print("Установите зависимости: pip install -r requirements_streamlit.txt")
        return False

def check_model():
    """Проверяет наличие обученной модели"""
    model_path = "runs/damage/weights/best.pt"
    if os.path.exists(model_path):
        print("✅ Модель найдена")
        return True
    else:
        print("⚠️  Модель не найдена. Сначала запустите обучение:")
        print("   python train_yolo.py")
        return False

def run_streamlit_app(app_file="app.py", port=8501):
    """Запускает Streamlit приложение"""
    print(f"🚀 Запускаем Streamlit приложение...")
    print(f"   Файл: {app_file}")
    print(f"   Порт: {port}")
    print(f"   URL: http://localhost:{port}")
    
    try:
        # Запускаем Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            app_file, 
            "--server.port", str(port),
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ]
        
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n⏹️  Приложение остановлено пользователем")
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при запуске: {e}")
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")

def main():
    """Главная функция"""
    print("=" * 60)
    print("🚗 ЗАПУСК STREAMLIT ПРИЛОЖЕНИЯ")
    print("=" * 60)
    
    # Выбор приложения
    print("\nВыберите версию приложения:")
    print("1. Базовая версия (app.py)")
    print("2. Продвинутая версия (app_advanced.py)")
    
    choice = input("\nВведите номер (1-2): ").strip()
    
    if choice == "1":
        app_file = "app.py"
    elif choice == "2":
        app_file = "app_advanced.py"
    else:
        print("❌ Неверный выбор!")
        return
    
    # Проверяем зависимости
    if not check_requirements():
        return
    
    # Проверяем модель
    model_exists = check_model()
    if not model_exists:
        print("\n⚠️  Приложение будет работать, но без анализа изображений")
        proceed = input("Продолжить? (y/n): ").strip().lower()
        if proceed != 'y':
            return
    
    # Запускаем приложение
    run_streamlit_app(app_file)

if __name__ == "__main__":
    main()
