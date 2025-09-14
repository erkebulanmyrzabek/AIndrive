#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для экспорта обученной модели YOLOv8 в формат ONNX
Автор: AI Assistant
"""

import os
import torch
from ultralytics import YOLO
import onnx
import onnxruntime as ort
import numpy as np

def check_onnx_installation():
    """Проверяет установку ONNX и ONNXRuntime"""
    try:
        import onnx
        import onnxruntime as ort
        print("✅ ONNX и ONNXRuntime установлены")
        print(f"   ONNX версия: {onnx.__version__}")
        print(f"   ONNXRuntime версия: {ort.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Ошибка импорта ONNX: {e}")
        print("   Установите: pip install onnx onnxruntime")
        return False

def export_to_onnx(model_path, output_dir="exports", imgsz=640):
    """Экспортирует модель YOLOv8 в формат ONNX"""
    print(f"🔄 Экспортируем модель в ONNX...")
    
    # Проверяем существование модели
    if not os.path.exists(model_path):
        print(f"❌ Модель не найдена: {model_path}")
        return False
    
    try:
        # Загружаем модель
        print(f"📥 Загружаем модель: {model_path}")
        model = YOLO(model_path)
        
        # Создаем выходную директорию
        os.makedirs(output_dir, exist_ok=True)
        
        # Экспортируем в ONNX
        print(f"⚙️  Параметры экспорта:")
        print(f"   Размер изображения: {imgsz}x{imgsz}")
        print(f"   Выходная директория: {output_dir}")
        
        # Выполняем экспорт
        exported_path = model.export(
            format='onnx',
            imgsz=imgsz,
            optimize=True,
            half=False,  # Используем FP32 для лучшей совместимости
            dynamic=False,  # Фиксированный размер входа
            simplify=True,  # Упрощение графа
            opset=11,  # Версия ONNX opset
            verbose=True
        )
        
        print(f"✅ Экспорт завершен успешно!")
        print(f"   ONNX модель: {exported_path}")
        
        return exported_path
        
    except Exception as e:
        print(f"❌ Ошибка при экспорте: {e}")
        return None

def validate_onnx_model(onnx_path, test_input_shape=(1, 3, 640, 640)):
    """Валидирует экспортированную ONNX модель"""
    print(f"🔍 Валидируем ONNX модель...")
    
    try:
        # Загружаем ONNX модель
        onnx_model = onnx.load(onnx_path)
        
        # Проверяем модель
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX модель валидна")
        
        # Тестируем с ONNXRuntime
        print("🧪 Тестируем с ONNXRuntime...")
        
        # Создаем сессию ONNXRuntime
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
        
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Получаем информацию о входе и выходе
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()
        
        print(f"   Вход: {input_info.name}, форма: {input_info.shape}")
        print(f"   Выходы: {len(output_info)}")
        for i, output in enumerate(output_info):
            print(f"     {i}: {output.name}, форма: {output.shape}")
        
        # Создаем тестовые данные
        test_input = np.random.randn(*test_input_shape).astype(np.float32)
        
        # Выполняем инференс
        outputs = session.run(None, {input_info.name: test_input})
        
        print(f"✅ ONNXRuntime тест прошел успешно")
        print(f"   Время инференса: ~{len(outputs)} выходов")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при валидации: {e}")
        return False

def get_model_info(model_path):
    """Получает информацию о модели"""
    try:
        model = YOLO(model_path)
        print(f"📊 Информация о модели:")
        print(f"   Архитектура: YOLOv8")
        print(f"   Размер файла: {os.path.getsize(model_path) / 1024**2:.1f} MB")
        
        # Пытаемся получить информацию о классах
        if hasattr(model, 'names'):
            print(f"   Классы: {list(model.names.values())}")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка при получении информации о модели: {e}")
        return False

def compare_model_sizes(original_path, onnx_path):
    """Сравнивает размеры оригинальной и ONNX моделей"""
    try:
        original_size = os.path.getsize(original_path) / 1024**2
        onnx_size = os.path.getsize(onnx_path) / 1024**2
        
        print(f"📏 Сравнение размеров моделей:")
        print(f"   PyTorch (.pt): {original_size:.1f} MB")
        print(f"   ONNX (.onnx): {onnx_size:.1f} MB")
        print(f"   Сжатие: {((original_size - onnx_size) / original_size * 100):.1f}%")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка при сравнении размеров: {e}")
        return False

def main():
    """Главная функция"""
    print("=" * 60)
    print("🔄 ЭКСПОРТ YOLOv8 В ONNX")
    print("=" * 60)
    
    # Проверяем установку ONNX
    if not check_onnx_installation():
        return
    
    # Путь к обученной модели
    model_path = "runs/damage/weights/best.pt"
    
    # Проверяем существование модели
    if not os.path.exists(model_path):
        print(f"❌ Модель не найдена: {model_path}")
        print("   Сначала запустите train_yolo.py для обучения модели")
        return
    
    # Получаем информацию о модели
    print("\n📋 Информация о модели:")
    get_model_info(model_path)
    
    # Экспортируем в ONNX
    print(f"\n🔄 Начинаем экспорт...")
    onnx_path = export_to_onnx(model_path, output_dir="exports", imgsz=640)
    
    if onnx_path is None:
        print("❌ Экспорт не удался!")
        return
    
    # Валидируем экспортированную модель
    print(f"\n🔍 Валидация...")
    if validate_onnx_model(onnx_path):
        print("✅ ONNX модель готова к использованию!")
    else:
        print("⚠️  ONNX модель создана, но валидация не прошла")
    
    # Сравниваем размеры
    print(f"\n📏 Анализ размеров:")
    compare_model_sizes(model_path, onnx_path)
    
    print(f"\n🎉 Экспорт завершен!")
    print(f"   ONNX модель: {onnx_path}")
    print(f"   Модель готова для развертывания в production!")

if __name__ == "__main__":
    main()
