#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Продвинутое Streamlit приложение для распознавания состояния автомобиля
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
    page_title="🚗 Advanced Car Damage Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стили
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
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
    
    .upload-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    
    .damage-badge {
        display: inline-block;
        padding: 0.6rem 1.2rem;
        border-radius: 30px;
        font-weight: bold;
        margin: 0.3rem;
        font-size: 0.9rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .clean { background: linear-gradient(135deg, #d4edda, #c3e6cb); color: #155724; }
    .dirty { background: linear-gradient(135deg, #fff3cd, #ffeaa7); color: #856404; }
    .dented { background: linear-gradient(135deg, #f8d7da, #f5c6cb); color: #721c24; }
    .scratched { background: linear-gradient(135deg, #cce5ff, #b3d9ff); color: #004085; }
    .broken { background: linear-gradient(135deg, #f5c6cb, #f1aeb5); color: #721c24; }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    
    .history-item {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Классы и их информация
CLASS_INFO = {
    0: {
        "name": "Чистый", 
        "color": "clean", 
        "emoji": "✨", 
        "description": "Автомобиль в отличном состоянии",
        "severity": "Нет",
        "action": "Поддерживайте текущее состояние"
    },
    1: {
        "name": "Грязный", 
        "color": "dirty", 
        "emoji": "💧", 
        "description": "Требуется мойка",
        "severity": "Низкая",
        "action": "Рекомендуется мойка"
    },
    2: {
        "name": "Вмятины", 
        "color": "dented", 
        "emoji": "🔨", 
        "description": "Механические повреждения",
        "severity": "Средняя",
        "action": "Обратитесь в автосервис"
    },
    3: {
        "name": "Царапины", 
        "color": "scratched", 
        "emoji": "🔪", 
        "description": "Поверхностные повреждения",
        "severity": "Средняя",
        "action": "Рассмотрите полировку"
    },
    4: {
        "name": "Разбитый", 
        "color": "broken", 
        "emoji": "💥", 
        "description": "Серьезные повреждения",
        "severity": "Высокая",
        "action": "Требуется срочный ремонт"
    }
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
    img_array = np.array(image)
    
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    return img_array

def draw_detections(image, boxes, scores, class_ids):
    """Отрисовывает детекции на изображении"""
    img_with_detections = image.copy()
    
    colors = {
        "clean": (0, 255, 0),
        "dirty": (0, 165, 255),
        "dented": (0, 0, 255),
        "scratched": (255, 0, 0),
        "broken": (128, 0, 128),
    }
    
    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        x1, y1, x2, y2 = map(int, box)
        class_info = CLASS_INFO.get(class_id, {"name": f"Class {class_id}", "color": "clean"})
        color = colors.get(class_info["color"], (255, 255, 255))
        
        # Рисуем прямоугольник
        cv2.rectangle(img_with_detections, (x1, y1), (x2, y2), color, 3)
        
        # Подпись
        label = f'{class_info["emoji"]} {class_info["name"]}: {score:.2f}'
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

def predict_damage(model, image, conf_threshold=0.5):
    """Предсказывает состояние автомобиля"""
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

def create_damage_chart(class_ids, scores):
    """Создает диаграмму повреждений"""
    if len(class_ids) == 0:
        return None
    
    # Подсчитываем количество каждого типа повреждений
    unique_classes, counts = np.unique(class_ids, return_counts=True)
    
    # Создаем данные для диаграммы
    damage_data = []
    for class_id, count in zip(unique_classes, counts):
        class_info = CLASS_INFO.get(class_id, {"name": f"Class {class_id}", "emoji": "❓"})
        damage_data.append({
            'Тип': f"{class_info['emoji']} {class_info['name']}",
            'Количество': count,
            'Серьезность': class_info['severity']
        })
    
    # Создаем диаграмму
    fig = px.bar(
        damage_data, 
        x='Тип', 
        y='Количество',
        color='Серьезность',
        title="Распределение типов повреждений",
        color_discrete_map={
            'Нет': '#28a745',
            'Низкая': '#ffc107', 
            'Средняя': '#fd7e14',
            'Высокая': '#dc3545'
        }
    )
    
    fig.update_layout(
        showlegend=True,
        height=400,
        font=dict(size=12)
    )
    
    return fig

def create_confidence_chart(scores):
    """Создает диаграмму уверенности"""
    if len(scores) == 0:
        return None
    
    fig = go.Figure(data=[
        go.Histogram(
            x=scores,
            nbinsx=20,
            marker_color='#667eea',
            opacity=0.7
        )
    ])
    
    fig.update_layout(
        title="Распределение уверенности детекций",
        xaxis_title="Уверенность",
        yaxis_title="Количество",
        height=300
    )
    
    return fig

def save_analysis_history(image_name, detections, timestamp):
    """Сохраняет историю анализов"""
    if 'analysis_history' not in st.session_state:
        st.session_state['analysis_history'] = []
    
    history_item = {
        'timestamp': timestamp,
        'image_name': image_name,
        'detections_count': len(detections['boxes']) if detections else 0,
        'detections': detections
    }
    
    st.session_state['analysis_history'].append(history_item)

def main():
    """Главная функция"""
    
    # Заголовок
    st.markdown('<h1 class="main-header">🚗 Advanced Car Damage Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Продвинутый анализ состояния автомобиля с помощью ИИ</p>', unsafe_allow_html=True)
    
    # Боковая панель
    with st.sidebar:
        st.markdown("## ⚙️ Настройки")
        
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
            st.info(f"Размер: {file_size:.1f} MB")
        else:
            st.error("❌ Модель не найдена")
        
        # Статистика сессии
        if 'analysis_history' in st.session_state:
            st.markdown("## 📈 Статистика сессии")
            total_analyses = len(st.session_state['analysis_history'])
            st.metric("Всего анализов", total_analyses)
            
            if total_analyses > 0:
                total_detections = sum(item['detections_count'] for item in st.session_state['analysis_history'])
                st.metric("Найдено повреждений", total_detections)
    
    # Основной контент с вкладками
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Анализ", "📊 Статистика", "📚 История", "ℹ️ О программе"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            st.markdown("### 📸 Загрузите фото автомобиля")
            
            uploaded_file = st.file_uploader(
                "Выберите изображение",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Поддерживаемые форматы: JPG, PNG, BMP"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Загруженное изображение", use_column_width=True)
                
                if st.button("🔍 Анализировать", type="primary", use_container_width=True):
                    with st.spinner("Анализируем изображение..."):
                        model = load_model()
                        
                        if model is not None:
                            img_array = preprocess_image(image)
                            boxes, scores, class_ids = predict_damage(model, img_array, conf_threshold)
                            
                            if len(boxes) > 0:
                                img_with_detections = draw_detections(img_array, boxes, scores, class_ids)
                                img_with_detections_rgb = cv2.cvtColor(img_with_detections, cv2.COLOR_BGR2RGB)
                                
                                st.session_state['result_image'] = img_with_detections_rgb
                                st.session_state['detections'] = {
                                    'boxes': boxes,
                                    'scores': scores,
                                    'class_ids': class_ids
                                }
                                
                                # Сохраняем в историю
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                save_analysis_history(uploaded_file.name, st.session_state['detections'], timestamp)
                                
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
                st.image(st.session_state['result_image'], caption="Результат анализа", use_column_width=True)
                
                if 'detections' in st.session_state and st.session_state['detections'] is not None:
                    detections = st.session_state['detections']
                    boxes = detections['boxes']
                    scores = detections['scores']
                    class_ids = detections['class_ids']
                    
                    # Метрики
                    col_metric1, col_metric2, col_metric3 = st.columns(3)
                    
                    with col_metric1:
                        st.metric("Найдено повреждений", len(boxes))
                    
                    with col_metric2:
                        avg_confidence = np.mean(scores) if len(scores) > 0 else 0
                        st.metric("Средняя уверенность", f"{avg_confidence:.2f}")
                    
                    with col_metric3:
                        max_confidence = np.max(scores) if len(scores) > 0 else 0
                        st.metric("Максимальная уверенность", f"{max_confidence:.2f}")
                    
                    # Детали
                    st.markdown("#### 🔍 Детали повреждений")
                    
                    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                        class_info = CLASS_INFO.get(class_id, {"name": f"Class {class_id}", "color": "clean"})
                        
                        with st.expander(f"{class_info['emoji']} {class_info['name']} (уверенность: {score:.2f})"):
                            st.markdown(f"**Описание:** {class_info['description']}")
                            st.markdown(f"**Серьезность:** {class_info['severity']}")
                            st.markdown(f"**Рекомендация:** {class_info['action']}")
                            st.markdown(f"**Координаты:** x1={box[0]:.0f}, y1={box[1]:.0f}, x2={box[2]:.0f}, y2={box[3]:.0f}")
                            st.markdown(f"**Размер:** {box[2]-box[0]:.0f} x {box[3]-box[1]:.0f} пикселей")
            else:
                st.info("👈 Загрузите изображение и нажмите 'Анализировать'")
    
    with tab2:
        st.markdown("### 📊 Статистика и визуализация")
        
        if 'detections' in st.session_state and st.session_state['detections'] is not None:
            detections = st.session_state['detections']
            class_ids = detections['class_ids']
            scores = detections['scores']
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Диаграмма типов повреждений
                damage_chart = create_damage_chart(class_ids, scores)
                if damage_chart:
                    st.plotly_chart(damage_chart, use_container_width=True)
            
            with col_chart2:
                # Диаграмма уверенности
                confidence_chart = create_confidence_chart(scores)
                if confidence_chart:
                    st.plotly_chart(confidence_chart, use_container_width=True)
            
            # Сводная таблица
            st.markdown("#### 📋 Сводная таблица")
            
            if len(class_ids) > 0:
                unique_classes, counts = np.unique(class_ids, return_counts=True)
                
                summary_data = []
                for class_id, count in zip(unique_classes, counts):
                    class_info = CLASS_INFO.get(class_id, {"name": f"Class {class_id}", "emoji": "❓"})
                    class_scores = scores[class_ids == class_id]
                    avg_score = np.mean(class_scores) if len(class_scores) > 0 else 0
                    
                    summary_data.append({
                        'Тип': f"{class_info['emoji']} {class_info['name']}",
                        'Количество': count,
                        'Средняя уверенность': f"{avg_score:.2f}",
                        'Серьезность': class_info['severity']
                    })
                
                import pandas as pd
                df = pd.DataFrame(summary_data)
                st.dataframe(df, use_container_width=True)
        else:
            st.info("👈 Выполните анализ для просмотра статистики")
    
    with tab3:
        st.markdown("### 📚 История анализов")
        
        if 'analysis_history' in st.session_state and len(st.session_state['analysis_history']) > 0:
            history = st.session_state['analysis_history']
            
            # Сортировка по времени (новые сверху)
            history_sorted = sorted(history, key=lambda x: x['timestamp'], reverse=True)
            
            for i, item in enumerate(history_sorted):
                with st.expander(f"📸 {item['image_name']} - {item['timestamp']}"):
                    st.markdown(f"**Время анализа:** {item['timestamp']}")
                    st.markdown(f"**Найдено повреждений:** {item['detections_count']}")
                    
                    if item['detections'] and item['detections_count'] > 0:
                        detections = item['detections']
                        class_ids = detections['class_ids']
                        scores = detections['scores']
                        
                        # Показываем типы повреждений
                        unique_classes, counts = np.unique(class_ids, return_counts=True)
                        for class_id, count in zip(unique_classes, counts):
                            class_info = CLASS_INFO.get(class_id, {"name": f"Class {class_id}", "emoji": "❓"})
                            st.markdown(f"- {class_info['emoji']} {class_info['name']}: {count} шт.")
            
            # Кнопка очистки истории
            if st.button("🗑️ Очистить историю", type="secondary"):
                st.session_state['analysis_history'] = []
                st.rerun()
        else:
            st.info("📝 История анализов пуста")
    
    with tab4:
        st.markdown("### ℹ️ О программе")
        
        st.markdown("""
        #### 🚗 Car Damage Detection
        
        Это приложение использует искусственный интеллект для анализа состояния автомобиля и выявления различных типов повреждений.
        
        #### 🎯 Возможности:
        - **Автоматическое распознавание** 5 типов состояний автомобиля
        - **Высокая точность** благодаря модели YOLOv8
        - **Интерактивный интерфейс** с визуализацией результатов
        - **Детальная статистика** и рекомендации
        - **История анализов** для отслеживания изменений
        
        #### 🏷️ Классы состояний:
        """)
        
        for class_id, info in CLASS_INFO.items():
            st.markdown(f"- **{info['emoji']} {info['name']}**: {info['description']}")
        
        st.markdown("""
        #### 🛠️ Технологии:
        - **YOLOv8** - современная архитектура для детекции объектов
        - **Streamlit** - веб-интерфейс
        - **OpenCV** - обработка изображений
        - **PyTorch** - машинное обучение
        
        #### 📞 Поддержка:
        При возникновении проблем обратитесь к документации или создайте issue в репозитории проекта.
        """)
    
    # Футер
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p>🚗 Advanced Car Damage Detection | Powered by YOLOv8 | Made with ❤️ and Streamlit</p>
            <p style='font-size: 0.8rem; margin-top: 1rem;'>
                Версия 2.0 | © 2024 | Все права защищены
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
