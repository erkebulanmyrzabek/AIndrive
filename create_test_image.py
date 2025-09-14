#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для создания тестового изображения с автомобилем
Автор: AI Assistant
"""

import cv2
import numpy as np
import os

def create_test_car_image(output_path="test_images/car1.jpg", width=800, height=600):
    """
    Создает тестовое изображение с автомобилем для демонстрации
    
    Args:
        output_path: путь для сохранения изображения
        width: ширина изображения
        height: высота изображения
    """
    print(f"🎨 Создаем тестовое изображение автомобиля...")
    
    # Создаем белый фон
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Рисуем автомобиль (простая схема)
    car_color = (100, 100, 100)  # Серый цвет
    
    # Основной корпус автомобиля
    car_body = np.array([
        [width//4, height//2 + 50],
        [width//4 + 200, height//2 + 50],
        [width//4 + 250, height//2 + 20],
        [width//4 + 300, height//2 + 20],
        [width//4 + 350, height//2 + 50],
        [width//4 + 400, height//2 + 50],
        [width//4 + 400, height//2 + 100],
        [width//4, height//2 + 100]
    ], np.int32)
    
    cv2.fillPoly(image, [car_body], car_color)
    
    # Добавляем окна
    window_color = (200, 200, 255)  # Светло-голубой
    
    # Переднее окно
    front_window = np.array([
        [width//4 + 250, height//2 + 30],
        [width//4 + 300, height//2 + 30],
        [width//4 + 300, height//2 + 60],
        [width//4 + 250, height//2 + 60]
    ], np.int32)
    cv2.fillPoly(image, [front_window], window_color)
    
    # Заднее окно
    rear_window = np.array([
        [width//4 + 300, height//2 + 30],
        [width//4 + 350, height//2 + 30],
        [width//4 + 350, height//2 + 60],
        [width//4 + 300, height//2 + 60]
    ], np.int32)
    cv2.fillPoly(image, [rear_window], window_color)
    
    # Колеса
    wheel_color = (50, 50, 50)  # Темно-серый
    
    # Переднее колесо
    cv2.circle(image, (width//4 + 80, height//2 + 100), 30, wheel_color, -1)
    cv2.circle(image, (width//4 + 80, height//2 + 100), 20, (100, 100, 100), -1)
    
    # Заднее колесо
    cv2.circle(image, (width//4 + 320, height//2 + 100), 30, wheel_color, -1)
    cv2.circle(image, (width//4 + 320, height//2 + 100), 20, (100, 100, 100), -1)
    
    # Добавляем "грязь" на автомобиль
    dirt_color = (139, 69, 19)  # Коричневый
    
    # Грязные пятна
    cv2.circle(image, (width//4 + 150, height//2 + 70), 15, dirt_color, -1)
    cv2.circle(image, (width//4 + 200, height//2 + 80), 12, dirt_color, -1)
    cv2.circle(image, (width//4 + 180, height//2 + 90), 10, dirt_color, -1)
    
    # Добавляем "царапины"
    scratch_color = (0, 0, 255)  # Красный
    
    # Горизонтальные царапины
    cv2.line(image, (width//4 + 100, height//2 + 60), (width//4 + 180, height//2 + 60), scratch_color, 2)
    cv2.line(image, (width//4 + 120, height//2 + 75), (width//4 + 200, height//2 + 75), scratch_color, 2)
    
    # Добавляем "вмятину"
    dent_color = (0, 0, 150)  # Темно-красный
    
    # Вмятина на крыле
    cv2.ellipse(image, (width//4 + 50, height//2 + 80), (25, 15), 0, 0, 360, dent_color, -1)
    cv2.ellipse(image, (width//4 + 50, height//2 + 80), (20, 10), 0, 0, 360, car_color, -1)
    
    # Добавляем "разбитую" часть
    broken_color = (0, 0, 0)  # Черный
    
    # Трещины на лобовом стекле
    cv2.line(image, (width//4 + 260, height//2 + 40), (width//4 + 290, height//2 + 50), broken_color, 3)
    cv2.line(image, (width//4 + 270, height//2 + 35), (width//4 + 285, height//2 + 55), broken_color, 2)
    
    # Добавляем тень под автомобилем
    shadow_color = (200, 200, 200)  # Светло-серый
    cv2.ellipse(image, (width//2, height//2 + 120), (200, 30), 0, 0, 360, shadow_color, -1)
    
    # Добавляем текст с информацией
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Test Car Image - Multiple Damage Types"
    text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
    text_x = (width - text_size[0]) // 2
    text_y = 30
    
    cv2.putText(image, text, (text_x, text_y), font, 0.7, (0, 0, 0), 2)
    
    # Добавляем легенду
    legend_y = height - 100
    legend_items = [
        ("Clean areas", (0, 255, 0)),
        ("Dirty spots", (139, 69, 19)),
        ("Scratches", (0, 0, 255)),
        ("Dent", (0, 0, 150)),
        ("Broken glass", (0, 0, 0))
    ]
    
    for i, (label, color) in enumerate(legend_items):
        y_pos = legend_y + i * 15
        cv2.circle(image, (20, y_pos), 5, color, -1)
        cv2.putText(image, label, (35, y_pos + 5), font, 0.4, (0, 0, 0), 1)
    
    # Создаем директорию если не существует
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Сохраняем изображение
    cv2.imwrite(output_path, image)
    print(f"✅ Тестовое изображение создано: {output_path}")
    print(f"   Размер: {width}x{height}")
    print(f"   Содержит: чистые области, грязь, царапины, вмятины, разбитое стекло")

def create_clean_car_image(output_path="test_images/car_clean.jpg", width=800, height=600):
    """Создает изображение чистого автомобиля"""
    print(f"🎨 Создаем изображение чистого автомобиля...")
    
    # Создаем белый фон
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Рисуем чистый автомобиль
    car_color = (0, 100, 200)  # Синий цвет
    
    # Основной корпус автомобиля
    car_body = np.array([
        [width//4, height//2 + 50],
        [width//4 + 200, height//2 + 50],
        [width//4 + 250, height//2 + 20],
        [width//4 + 300, height//2 + 20],
        [width//4 + 350, height//2 + 50],
        [width//4 + 400, height//2 + 50],
        [width//4 + 400, height//2 + 100],
        [width//4, height//2 + 100]
    ], np.int32)
    
    cv2.fillPoly(image, [car_body], car_color)
    
    # Добавляем блики (чистый автомобиль)
    highlight_color = (100, 150, 255)  # Светло-синий
    cv2.ellipse(image, (width//4 + 150, height//2 + 70), (60, 20), 0, 0, 360, highlight_color, -1)
    
    # Окна
    window_color = (200, 200, 255)
    front_window = np.array([
        [width//4 + 250, height//2 + 30],
        [width//4 + 300, height//2 + 30],
        [width//4 + 300, height//2 + 60],
        [width//4 + 250, height//2 + 60]
    ], np.int32)
    cv2.fillPoly(image, [front_window], window_color)
    
    rear_window = np.array([
        [width//4 + 300, height//2 + 30],
        [width//4 + 350, height//2 + 30],
        [width//4 + 350, height//2 + 60],
        [width//4 + 300, height//2 + 60]
    ], np.int32)
    cv2.fillPoly(image, [rear_window], window_color)
    
    # Колеса
    wheel_color = (50, 50, 50)
    cv2.circle(image, (width//4 + 80, height//2 + 100), 30, wheel_color, -1)
    cv2.circle(image, (width//4 + 80, height//2 + 100), 20, (100, 100, 100), -1)
    cv2.circle(image, (width//4 + 320, height//2 + 100), 30, wheel_color, -1)
    cv2.circle(image, (width//4 + 320, height//2 + 100), 20, (100, 100, 100), -1)
    
    # Добавляем текст
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Clean Car - No Damage"
    text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
    text_x = (width - text_size[0]) // 2
    text_y = 30
    
    cv2.putText(image, text, (text_x, text_y), font, 0.7, (0, 0, 0), 2)
    
    # Сохраняем
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"✅ Чистое изображение создано: {output_path}")

def main():
    """Главная функция"""
    print("=" * 60)
    print("🎨 СОЗДАНИЕ ТЕСТОВЫХ ИЗОБРАЖЕНИЙ")
    print("=" * 60)
    
    # Создаем изображение с различными типами повреждений
    create_test_car_image("test_images/car1.jpg")
    
    # Создаем изображение чистого автомобиля
    create_clean_car_image("test_images/car_clean.jpg")
    
    print(f"\n🎉 Тестовые изображения созданы!")
    print(f"   Теперь можно запустить detect.py для тестирования модели")

if __name__ == "__main__":
    main()
