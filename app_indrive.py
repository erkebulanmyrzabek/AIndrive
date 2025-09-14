#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit приложение для inDrive кейса
Определение состояния автомобиля: чистота + целостность
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
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Настройка страницы
st.set_page_config(
    page_title="🚗 inDrive Car Condition Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стили для inDrive
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #FF6B35 0%, #F7931E 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.3rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    .indrive-card {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(255, 107, 53, 0.3);
    }
    
    .priority-critical {
        background: linear-gradient(135deg, #dc3545, #c82333);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border-left: 5px solid #dc3545;
    }
    
    .priority-high {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border-left: 5px solid #28a745;
    }
    
    .priority-medium {
        background: linear-gradient(135deg, #ffc107, #fd7e14);
        color: #212529;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border-left: 5px solid #ffc107;
    }
    
    .priority-low {
        background: linear-gradient(135deg, #6c757d, #495057);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border-left: 5px solid #6c757d;
    }
    
    .upload-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        border: 2px solid #FF6B35;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 5px 15px rgba(255, 107, 53, 0.3);
    }
    
    .recommendation-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #FF6B35;
    }
</style>
""", unsafe_allow_html=True)

# Классы для inDrive кейса
INDRIVE_CLASSES = {
    0: {
        "name": "clean_intact",
        "display_name": "Чистый и целый",
        "emoji": "✨",
        "description": "Автомобиль в отличном состоянии",
        "priority": "high",
        "action": "✅ Можно принимать заказы",
        "color": "#28a745"
    },
    1: {
        "name": "clean_damaged", 
        "display_name": "Чистый, но поврежденный",
        "emoji": "🔧",
        "description": "Есть царапины или вмятины",
        "priority": "medium",
        "action": "⚠️ Требует внимания водителя",
        "color": "#ffc107"
    },
    2: {
        "name": "dirty_intact",
        "display_name": "Грязный, но целый", 
        "emoji": "💧",
        "description": "Нужна мойка",
        "priority": "medium",
        "action": "🧽 Рекомендуется мойка",
        "color": "#17a2b8"
    },
    3: {
        "name": "dirty_damaged",
        "display_name": "Грязный и поврежденный",
        "emoji": "🔨",
        "description": "Требует мойки и ремонта",
        "priority": "low",
        "action": "🚫 Не рекомендуется для заказов",
        "color": "#6c757d"
    },
    4: {
        "name": "very_dirty",
        "display_name": "Очень грязный",
        "emoji": "💩",
        "description": "Критическое состояние чистоты",
        "priority": "low",
        "action": "🚫 Заблокировать до мойки",
        "color": "#6c757d"
    },
    5: {
        "name": "severely_damaged",
        "display_name": "Сильно поврежденный",
        "emoji": "💥",
        "description": "Небезопасен для перевозки",
        "priority": "critical",
        "action": "🚨 КРИТИЧНО: Заблокировать немедленно",
        "color": "#dc3545"
    }
}

@st.cache_resource
def load_model():
    """Загружает модель YOLOv8 для inDrive"""
    model_path = "indrive_runs/car_condition/weights/best.pt"
    
    if not os.path.exists(model_path):
        st.error("❌ Модель не найдена! Сначала запустите обучение: `python train_indrive.py`")
        return None
    
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Ошибка при загрузке модели: {e}")
        return None

def preprocess_image(image):
    """Предобработка изображения"""
    img_array = np.array(image)
    
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    return img_array

def draw_indrive_detections(image, boxes, scores, class_ids):
    """Отрисовывает детекции в стиле inDrive"""
    img_with_detections = image.copy()
    
    colors = {
        "clean_intact": (0, 255, 0),      # зеленый
        "clean_damaged": (255, 255, 0),   # желтый
        "dirty_intact": (0, 255, 255),    # голубой
        "dirty_damaged": (128, 128, 128), # серый
        "very_dirty": (0, 0, 128),        # темно-синий
        "severely_damaged": (0, 0, 255),  # красный
    }
    
    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        x1, y1, x2, y2 = map(int, box)
        class_info = INDRIVE_CLASSES.get(class_id, {"name": f"class_{class_id}", "display_name": f"Class {class_id}"})
        color = colors.get(class_info["name"], (255, 255, 255))
        
        # Рисуем прямоугольник
        cv2.rectangle(img_with_detections, (x1, y1), (x2, y2), color, 3)
        
        # Подпись
        label = f'{class_info["emoji"]} {class_info["display_name"]}: {score:.2f}'
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        
        # Фон для текста
        cv2.rectangle(
            img_with_detections,
            (x1, y1 - text_height - baseline - 10),
            (x1 + text_width + 10, y1),
            color,
            -1
        )
        
        # Текст
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

def predict_indrive_condition(model, image, conf_threshold=0.5):
    """Предсказывает состояние автомобиля для inDrive"""
    try:
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

def get_indrive_recommendation(class_ids, scores):
    """Получает рекомендации для inDrive на основе результатов"""
    if len(class_ids) == 0:
        return "Не удалось определить состояние автомобиля"
    
    # Находим наиболее критичный класс
    priorities = {"critical": 4, "low": 3, "medium": 2, "high": 1}
    max_priority = 0
    critical_class = None
    
    for class_id in class_ids:
        class_info = INDRIVE_CLASSES.get(class_id, {"priority": "medium"})
        priority_level = priorities.get(class_info["priority"], 2)
        
        if priority_level > max_priority:
            max_priority = priority_level
            critical_class = class_id
    
    if critical_class is not None:
        class_info = INDRIVE_CLASSES[critical_class]
        return {
            "status": class_info["action"],
            "priority": class_info["priority"],
            "description": class_info["description"],
            "class_name": class_info["display_name"]
        }
    
    return "Состояние не определено"

def create_indrive_dashboard(class_ids, scores):
    """Создает дашборд для inDrive"""
    if len(class_ids) == 0:
        return None
    
    # Подсчитываем классы
    unique_classes, counts = np.unique(class_ids, return_counts=True)
    
    # Создаем данные для дашборда
    dashboard_data = []
    for class_id, count in zip(unique_classes, counts):
        class_info = INDRIVE_CLASSES.get(class_id, {"display_name": f"Class {class_id}", "priority": "medium"})
        class_scores = scores[class_ids == class_id]
        avg_score = np.mean(class_scores) if len(class_scores) > 0 else 0
        
        dashboard_data.append({
            'Класс': f"{class_info['emoji']} {class_info['display_name']}",
            'Количество': count,
            'Средняя уверенность': avg_score,
            'Приоритет': class_info['priority'],
            'Цвет': class_info['color']
        })
    
    # Создаем диаграмму
    fig = px.bar(
        dashboard_data,
        x='Класс',
        y='Количество',
        color='Приоритет',
        title="Анализ состояния автомобиля для inDrive",
        color_discrete_map={
            'critical': '#dc3545',
            'low': '#6c757d',
            'medium': '#ffc107',
            'high': '#28a745'
        }
    )
    
    fig.update_layout(
        showlegend=True,
        height=400,
        font=dict(size=12)
    )
    
    return fig

def main():
    """Главная функция"""
    
    # Заголовок
    st.markdown('<h1 class="main-header">🚗 inDrive Car Condition Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Определение состояния автомобиля для повышения качества сервиса</p>', unsafe_allow_html=True)
    
    # Боковая панель
    with st.sidebar:
        st.markdown("## ⚙️ Настройки анализа")
        
        conf_threshold = st.slider(
            "Порог уверенности",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Минимальная уверенность для показа детекций"
        )
        
        # Информация о модели
        st.markdown("## 📊 Статус модели")
        model_path = "indrive_runs/car_condition/weights/best.pt"
        if os.path.exists(model_path):
            st.success("✅ Модель загружена")
            file_size = os.path.getsize(model_path) / (1024 * 1024)
            st.info(f"Размер: {file_size:.1f} MB")
        else:
            st.error("❌ Модель не найдена")
        
        # Классы состояний
        st.markdown("## 🏷️ Классы состояний")
        for class_id, info in INDRIVE_CLASSES.items():
            priority_color = {
                "critical": "🔴",
                "low": "⚫", 
                "medium": "🟡",
                "high": "🟢"
            }.get(info["priority"], "⚪")
            
            st.markdown(f"**{priority_color} {info['emoji']} {info['display_name']}**")
            st.caption(f"{info['description']}")
    
    # Основной контент
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### 📸 Загрузите фото автомобиля")
        
        uploaded_file = st.file_uploader(
            "Выберите изображение автомобиля",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Поддерживаемые форматы: JPG, PNG, BMP"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Загруженное изображение", use_column_width=True)
            
            if st.button("🔍 Анализировать состояние", type="primary", use_container_width=True):
                with st.spinner("Анализируем состояние автомобиля..."):
                    model = load_model()
                    
                    if model is not None:
                        img_array = preprocess_image(image)
                        boxes, scores, class_ids = predict_indrive_condition(model, img_array, conf_threshold)
                        
                        if len(boxes) > 0:
                            img_with_detections = draw_indrive_detections(img_array, boxes, scores, class_ids)
                            img_with_detections_rgb = cv2.cvtColor(img_with_detections, cv2.COLOR_BGR2RGB)
                            
                            st.session_state['result_image'] = img_with_detections_rgb
                            st.session_state['detections'] = {
                                'boxes': boxes,
                                'scores': scores,
                                'class_ids': class_ids
                            }
                            
                            st.success("✅ Анализ завершен!")
                        else:
                            st.warning("⚠️ Состояние автомобиля не определено")
                            st.session_state['result_image'] = None
                            st.session_state['detections'] = None
        else:
            st.info("👆 Загрузите изображение для анализа")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Результаты анализа")
        
        if 'result_image' in st.session_state and st.session_state['result_image'] is not None:
            st.image(st.session_state['result_image'], caption="Результат анализа", use_column_width=True)
            
            if 'detections' in st.session_state and st.session_state['detections'] is not None:
                detections = st.session_state['detections']
                boxes = detections['boxes']
                scores = detections['scores']
                class_ids = detections['class_ids']
                
                # Получаем рекомендацию для inDrive
                recommendation = get_indrive_recommendation(class_ids, scores)
                
                # Показываем рекомендацию
                st.markdown("#### 🎯 Рекомендация для inDrive")
                
                if isinstance(recommendation, dict):
                    priority_class = recommendation["priority"]
                    
                    if priority_class == "critical":
                        st.markdown(f'<div class="priority-critical">', unsafe_allow_html=True)
                        st.markdown(f"🚨 **{recommendation['status']}**")
                        st.markdown(f"**Класс:** {recommendation['class_name']}")
                        st.markdown(f"**Описание:** {recommendation['description']}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    elif priority_class == "high":
                        st.markdown(f'<div class="priority-high">', unsafe_allow_html=True)
                        st.markdown(f"✅ **{recommendation['status']}**")
                        st.markdown(f"**Класс:** {recommendation['class_name']}")
                        st.markdown(f"**Описание:** {recommendation['description']}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    elif priority_class == "medium":
                        st.markdown(f'<div class="priority-medium">', unsafe_allow_html=True)
                        st.markdown(f"⚠️ **{recommendation['status']}**")
                        st.markdown(f"**Класс:** {recommendation['class_name']}")
                        st.markdown(f"**Описание:** {recommendation['description']}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="priority-low">', unsafe_allow_html=True)
                        st.markdown(f"🚫 **{recommendation['status']}**")
                        st.markdown(f"**Класс:** {recommendation['class_name']}")
                        st.markdown(f"**Описание:** {recommendation['description']}")
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info(recommendation)
                
                # Метрики
                col_metric1, col_metric2, col_metric3 = st.columns(3)
                
                with col_metric1:
                    st.metric("Найдено проблем", len(boxes))
                
                with col_metric2:
                    avg_confidence = np.mean(scores) if len(scores) > 0 else 0
                    st.metric("Средняя уверенность", f"{avg_confidence:.2f}")
                
                with col_metric3:
                    max_confidence = np.max(scores) if len(scores) > 0 else 0
                    st.metric("Максимальная уверенность", f"{max_confidence:.2f}")
                
                # Дашборд
                st.markdown("#### 📈 Дашборд анализа")
                dashboard_fig = create_indrive_dashboard(class_ids, scores)
                if dashboard_fig:
                    st.plotly_chart(dashboard_fig, use_container_width=True)
                
                # Детали
                st.markdown("#### 🔍 Детали обнаруженных проблем")
                
                for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                    class_info = INDRIVE_CLASSES.get(class_id, {"display_name": f"Class {class_id}", "priority": "medium"})
                    
                    with st.expander(f"{class_info['emoji']} {class_info['display_name']} (уверенность: {score:.2f})"):
                        st.markdown(f"**Описание:** {class_info['description']}")
                        st.markdown(f"**Приоритет:** {class_info['priority']}")
                        st.markdown(f"**Рекомендация:** {class_info['action']}")
                        st.markdown(f"**Координаты:** x1={box[0]:.0f}, y1={box[1]:.0f}, x2={box[2]:.0f}, y2={box[3]:.0f}")
                        st.markdown(f"**Размер:** {box[2]-box[0]:.0f} x {box[3]-box[1]:.0f} пикселей")
        else:
            st.info("👈 Загрузите изображение и нажмите 'Анализировать состояние'")
    
    # Футер
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p>🚗 inDrive Car Condition Detection | Powered by YOLOv8 | Made with ❤️ for inDrive</p>
            <p style='font-size: 0.8rem; margin-top: 1rem;'>
                Версия 1.0 | © 2024 inDrive | Все права защищены
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
