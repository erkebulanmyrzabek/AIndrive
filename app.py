#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit приложение для распознавания состояния автомобиля
Автор: AI Assistant
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from ultralytics import YOLO
import time

# Настройка страницы
st.set_page_config(
    page_title="🚗 Car Damage Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стили
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    
    .upload-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .damage-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: bold;
        margin: 0.25rem;
    }
    
    .clean { background-color: #d4edda; color: #155724; }
    .dirty { background-color: #fff3cd; color: #856404; }
    .dented { background-color: #f8d7da; color: #721c24; }
    .scratched { background-color: #cce5ff; color: #004085; }
    .broken { background-color: #f5c6cb; color: #721c24; }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Классы и их цвета
CLASS_INFO = {
    0: {"name": "Чистый", "color": "clean", "emoji": "✨", "description": "Автомобиль в отличном состоянии"},
    1: {"name": "Грязный", "color": "dirty", "emoji": "💧", "description": "Требуется мойка"},
    2: {"name": "Вмятины", "color": "dented", "emoji": "🔨", "description": "Механические повреждения"},
    3: {"name": "Царапины", "color": "scratched", "emoji": "🔪", "description": "Поверхностные повреждения"},
    4: {"name": "Разбитый", "color": "broken", "emoji": "💥", "description": "Серьезные повреждения"}
}

@st.cache_resource
def load_model():
    """Загружает модель YOLOv8"""
    model_path = "runs/damage/weights/best.pt"
    
    if not os.path.exists(model_path):
        st.error("❌ Модель не найдена! Сначала запустите обучение: `python train_yolo.py`")
        return None
    
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Ошибка при загрузке модели: {e}")
        return None

def preprocess_image(image):
    """Предобработка изображения"""
    # Конвертируем PIL в OpenCV
    img_array = np.array(image)
    
    # Конвертируем RGB в BGR для OpenCV
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    return img_array

def draw_detections(image, boxes, scores, class_ids):
    """Отрисовывает детекции на изображении"""
    img_with_detections = image.copy()
    
    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        # Получаем координаты bbox
        x1, y1, x2, y2 = map(int, box)
        
        # Получаем информацию о классе
        class_info = CLASS_INFO.get(class_id, {"name": f"Class {class_id}", "color": "clean"})
        
        # Цвета для bbox
        colors = {
            "clean": (0, 255, 0),      # зеленый
            "dirty": (0, 165, 255),    # оранжевый
            "dented": (0, 0, 255),     # красный
            "scratched": (255, 0, 0),  # синий
            "broken": (128, 0, 128),   # фиолетовый
        }
        
        color = colors.get(class_info["color"], (255, 255, 255))
        
        # Рисуем прямоугольник
        cv2.rectangle(img_with_detections, (x1, y1), (x2, y2), color, 3)
        
        # Подготавливаем текст
        label = f'{class_info["name"]}: {score:.2f}'
        
        # Получаем размер текста
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        
        # Рисуем фон для текста
        cv2.rectangle(
            img_with_detections,
            (x1, y1 - text_height - baseline - 10),
            (x1 + text_width + 10, y1),
            color,
            -1
        )
        
        # Рисуем текст
        cv2.putText(
            img_with_detections,
            label,
            (x1 + 5, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
    
    return img_with_detections

def predict_damage(model, image, conf_threshold=0.5):
    """Предсказывает состояние автомобиля"""
    try:
        # Выполняем детекцию
        results = model(image, conf=conf_threshold, verbose=False)
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            return boxes, scores, class_ids
        else:
            return np.array([]), np.array([]), np.array([])
            
    except Exception as e:
        st.error(f"❌ Ошибка при предсказании: {e}")
        return np.array([]), np.array([]), np.array([])

def main():
    """Главная функция"""
    
    # Заголовок
    st.markdown('<h1 class="main-header">🚗 Car Damage Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Искусственный интеллект для анализа состояния автомобиля</p>', unsafe_allow_html=True)
    
    # Боковая панель
    with st.sidebar:
        st.markdown("## ⚙️ Настройки")
        
        # Порог уверенности
        conf_threshold = st.slider(
            "Порог уверенности",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Минимальная уверенность для показа детекций"
        )
        
        # Информация о модели
        st.markdown("## 📊 Информация о модели")
        model_path = "runs/damage/weights/best.pt"
        if os.path.exists(model_path):
            st.success("✅ Модель загружена")
            file_size = os.path.getsize(model_path) / (1024 * 1024)
            st.info(f"Размер модели: {file_size:.1f} MB")
        else:
            st.error("❌ Модель не найдена")
        
        # Классы
        st.markdown("## 🏷️ Классы состояний")
        for class_id, info in CLASS_INFO.items():
            st.markdown(f"**{info['emoji']} {info['name']}**")
            st.caption(info['description'])
    
    # Основной контент
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### 📸 Загрузите фото автомобиля")
        
        # Загрузка файла
        uploaded_file = st.file_uploader(
            "Выберите изображение",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Поддерживаемые форматы: JPG, PNG, BMP"
        )
        
        if uploaded_file is not None:
            # Показываем загруженное изображение
            image = Image.open(uploaded_file)
            st.image(image, caption="Загруженное изображение", use_column_width=True)
            
            # Кнопка анализа
            if st.button("🔍 Анализировать", type="primary", use_container_width=True):
                with st.spinner("Анализируем изображение..."):
                    # Загружаем модель
                    model = load_model()
                    
                    if model is not None:
                        # Предобработка
                        img_array = preprocess_image(image)
                        
                        # Предсказание
                        boxes, scores, class_ids = predict_damage(model, img_array, conf_threshold)
                        
                        if len(boxes) > 0:
                            # Отрисовываем детекции
                            img_with_detections = draw_detections(img_array, boxes, scores, class_ids)
                            
                            # Конвертируем обратно в RGB для отображения
                            img_with_detections_rgb = cv2.cvtColor(img_with_detections, cv2.COLOR_BGR2RGB)
                            
                            # Сохраняем результат в session state
                            st.session_state['result_image'] = img_with_detections_rgb
                            st.session_state['detections'] = {
                                'boxes': boxes,
                                'scores': scores,
                                'class_ids': class_ids
                            }
                            
                            st.success("✅ Анализ завершен!")
                        else:
                            st.warning("⚠️ Повреждения не обнаружены")
                            st.session_state['result_image'] = None
                            st.session_state['detections'] = None
        else:
            st.info("👆 Загрузите изображение для анализа")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Результаты анализа")
        
        if 'result_image' in st.session_state and st.session_state['result_image'] is not None:
            # Показываем результат
            st.image(st.session_state['result_image'], caption="Результат анализа", use_column_width=True)
            
            # Показываем детекции
            if 'detections' in st.session_state and st.session_state['detections'] is not None:
                detections = st.session_state['detections']
                boxes = detections['boxes']
                scores = detections['scores']
                class_ids = detections['class_ids']
                
                # Статистика
                st.markdown("#### 📈 Статистика")
                col_metric1, col_metric2, col_metric3 = st.columns(3)
                
                with col_metric1:
                    st.metric("Найдено повреждений", len(boxes))
                
                with col_metric2:
                    avg_confidence = np.mean(scores) if len(scores) > 0 else 0
                    st.metric("Средняя уверенность", f"{avg_confidence:.2f}")
                
                with col_metric3:
                    max_confidence = np.max(scores) if len(scores) > 0 else 0
                    st.metric("Максимальная уверенность", f"{max_confidence:.2f}")
                
                # Детали детекций
                st.markdown("#### 🔍 Детали повреждений")
                
                for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                    class_info = CLASS_INFO.get(class_id, {"name": f"Class {class_id}", "color": "clean"})
                    
                    with st.expander(f"{class_info['emoji']} {class_info['name']} (уверенность: {score:.2f})"):
                        st.markdown(f"**Описание:** {class_info['description']}")
                        st.markdown(f"**Координаты:** x1={box[0]:.0f}, y1={box[1]:.0f}, x2={box[2]:.0f}, y2={box[3]:.0f}")
                        st.markdown(f"**Размер:** {box[2]-box[0]:.0f} x {box[3]-box[1]:.0f} пикселей")
                
                # Рекомендации
                st.markdown("#### 💡 Рекомендации")
                
                unique_classes = np.unique(class_ids)
                recommendations = []
                
                for class_id in unique_classes:
                    class_info = CLASS_INFO.get(class_id, {"name": f"Class {class_id}"})
                    
                    if class_id == 0:  # clean
                        recommendations.append("✅ Автомобиль в отличном состоянии!")
                    elif class_id == 1:  # dirty
                        recommendations.append("🧽 Рекомендуется мойка автомобиля")
                    elif class_id == 2:  # dented
                        recommendations.append("🔧 Обратитесь в автосервис для устранения вмятин")
                    elif class_id == 3:  # scratched
                        recommendations.append("🎨 Рассмотрите возможность полировки или покраски")
                    elif class_id == 4:  # broken
                        recommendations.append("⚠️ Требуется срочный ремонт в автосервисе")
                
                for rec in recommendations:
                    st.info(rec)
        
        else:
            st.info("👈 Загрузите изображение и нажмите 'Анализировать'")
    
    # Футер
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>🚗 Car Damage Detection | Powered by YOLOv8 | Made with ❤️ and Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
