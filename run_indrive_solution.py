#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Главный скрипт для запуска полного решения inDrive кейса
Автор: AI Assistant
"""

import subprocess
import sys
import os
import time

def print_banner():
    """Печатает баннер"""
    print("=" * 80)
    print("🚗 inDrive Car Condition Detection Solution")
    print("   Определение состояния автомобиля для повышения качества сервиса")
    print("=" * 80)

def check_dependencies():
    """Проверяет зависимости"""
    print("🔍 Проверяем зависимости...")
    
    required_packages = [
        'streamlit', 'ultralytics', 'opencv-python', 
        'torch', 'plotly', 'numpy', 'PIL'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  Отсутствуют пакеты: {', '.join(missing_packages)}")
        print("   Установите: pip install -r requirements_streamlit.txt")
        return False
    
    print("✅ Все зависимости установлены")
    return True

def create_demo_data():
    """Создает демонстрационные данные"""
    print("\n🎨 Создаем демонстрационные данные...")
    
    try:
        # Создаем структуру датасета
        subprocess.run([sys.executable, "create_dataset_structure.py"], check=True)
        
        # Создаем тестовые изображения
        subprocess.run([sys.executable, "create_test_image.py"], check=True)
        
        print("✅ Демонстрационные данные созданы")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при создании демо данных: {e}")
        return False

def train_model():
    """Обучает модель"""
    print("\n🤖 Обучаем модель...")
    
    model_path = "indrive_runs/car_condition/weights/best.pt"
    
    if os.path.exists(model_path):
        print("✅ Модель уже обучена")
        return True
    
    try:
        print("   Запускаем обучение (это может занять время)...")
        subprocess.run([sys.executable, "train_indrive.py"], check=True)
        
        if os.path.exists(model_path):
            print("✅ Модель обучена успешно")
            return True
        else:
            print("❌ Обучение не завершилось")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при обучении: {e}")
        return False

def create_presentation():
    """Создает презентационные материалы"""
    print("\n📊 Создаем презентационные материалы...")
    
    try:
        subprocess.run([sys.executable, "presentation_materials.py"], check=True)
        print("✅ Презентационные материалы созданы")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при создании материалов: {e}")
        return False

def run_demo():
    """Запускает демонстрацию"""
    print("\n🚀 Запускаем демонстрацию...")
    
    try:
        print("   Открываем Streamlit приложение...")
        print("   URL: http://localhost:8501")
        print("   Нажмите Ctrl+C для остановки")
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app_indrive.py",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n⏹️  Демонстрация остановлена")
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при запуске демо: {e}")

def show_menu():
    """Показывает меню"""
    print("\n📋 Выберите действие:")
    print("1. 🚀 Полный запуск (создание данных + обучение + демо)")
    print("2. 🎨 Только создание демо данных")
    print("3. 🤖 Только обучение модели")
    print("4. 🚀 Только запуск демо")
    print("5. 📊 Создать презентационные материалы")
    print("6. ❌ Выход")

def main():
    """Главная функция"""
    print_banner()
    
    # Проверяем зависимости
    if not check_dependencies():
        return
    
    while True:
        show_menu()
        
        choice = input("\nВведите номер (1-6): ").strip()
        
        if choice == "1":
            # Полный запуск
            print("\n🚀 Полный запуск решения...")
            
            if not create_demo_data():
                continue
                
            if not train_model():
                print("⚠️  Модель не обучена, но можно запустить демо с базовой моделью")
            
            create_presentation()
            run_demo()
            
        elif choice == "2":
            # Только создание данных
            create_demo_data()
            
        elif choice == "3":
            # Только обучение
            train_model()
            
        elif choice == "4":
            # Только демо
            run_demo()
            
        elif choice == "5":
            # Презентационные материалы
            create_presentation()
            
        elif choice == "6":
            # Выход
            print("👋 До свидания!")
            break
            
        else:
            print("❌ Неверный выбор!")
        
        input("\nНажмите Enter для продолжения...")

if __name__ == "__main__":
    main()
