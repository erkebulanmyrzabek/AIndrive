# 🚗 YOLOv8 для распознавания состояния автомобиля

Минимальный проект для обучения и использования YOLOv8 для распознавания состояния автомобиля (чистый, грязный, с вмятинами, царапинами, битый).

## 📋 Описание

Этот проект содержит полный pipeline для:
- Обучения модели YOLOv8 на распознавание состояния автомобиля
- Детекции состояния на новых изображениях
- Экспорта модели в формат ONNX для production

## 🎯 Классы состояний

Модель распознает 5 классов состояний автомобиля:
- **clean** - чистый автомобиль
- **dirty** - грязный автомобиль  
- **dented** - автомобиль с вмятинами
- **scratched** - автомобиль с царапинами
- **broken** - битый автомобиль

## 📁 Структура проекта

```
talpyn-dashboard/
├── requirements.txt          # Зависимости Python
├── data.yaml                # Конфигурация датасета
├── train_yolo.py            # Скрипт обучения модели
├── detect.py                # Скрипт детекции
├── export_onnx.py           # Экспорт в ONNX
├── README.md                # Документация
├── dataset/                 # Датасет (создать вручную)
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
├── test_images/             # Тестовые изображения
│   └── car1.jpg
├── runs/                    # Результаты обучения
│   └── damage/
│       └── weights/
│           ├── best.pt
│           └── last.pt
└── exports/                 # Экспортированные модели
    └── best.onnx
```

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
# Создайте виртуальное окружение (рекомендуется)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# Установите зависимости
pip install -r requirements.txt
```

### 2. Подготовка датасета

Создайте структуру датасета согласно `data.yaml`:

```bash
mkdir -p dataset/images/{train,val,test}
mkdir -p dataset/labels/{train,val,test}
```

**Формат аннотаций YOLO:**
- Каждое изображение должно иметь соответствующий `.txt` файл с аннотациями
- Формат: `class_id center_x center_y width height` (нормализованные координаты 0-1)
- Пример: `2 0.5 0.5 0.3 0.4` (класс 2, центр в середине, размер 30%x40%)

### 3. Обучение модели

```bash
python train_yolo.py
```

**Параметры обучения:**
- Эпохи: 50
- Размер изображений: 640x640
- Батч: 16 (GPU) / 8 (CPU)
- Устройство: автоматическое определение GPU/CPU

**Результаты:**
- Модель сохраняется в `runs/damage/weights/`
- `best.pt` - лучшая модель по валидации
- `last.pt` - последняя эпоха

### 4. Детекция

```bash
python detect.py
```

**Что делает:**
- Загружает лучшую модель `runs/damage/weights/best.pt`
- Обрабатывает `test_images/car1.jpg`
- Сохраняет результат в `runs/damage_predict/`
- Рисует bbox и подписи классов

### 5. Экспорт в ONNX

```bash
python export_onnx.py
```

**Результат:**
- ONNX модель в `exports/best.onnx`
- Валидация модели
- Сравнение размеров файлов

## 📊 Использование

### Обучение с кастомными параметрами

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='cuda'
)
```

### Детекция на новом изображении

```python
from ultralytics import YOLO
import cv2

# Загружаем модель
model = YOLO('runs/damage/weights/best.pt')

# Детекция
results = model('path/to/image.jpg', conf=0.5)

# Обработка результатов
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        cls = int(box.cls[0])
        print(f"Класс: {cls}, Уверенность: {conf:.2f}")
```

### Использование ONNX модели

```python
import onnxruntime as ort
import numpy as np

# Загружаем ONNX модель
session = ort.InferenceSession('exports/best.onnx')

# Подготавливаем входные данные
input_data = np.random.randn(1, 3, 640, 640).astype(np.float32)

# Инференс
outputs = session.run(None, {'images': input_data})
```

## ⚙️ Настройка

### Изменение классов

Отредактируйте `data.yaml`:

```yaml
nc: 5  # количество классов
names:
  0: clean
  1: dirty
  2: dented
  3: scratched
  4: broken
```

### Изменение параметров обучения

Отредактируйте `train_yolo.py`, секция `training_args`:

```python
training_args = {
    'epochs': 100,        # больше эпох
    'imgsz': 1280,        # большее разрешение
    'batch': 32,          # больший батч
    'lr0': 0.001,         # меньшая скорость обучения
    # ... другие параметры
}
```

### Изменение порога детекции

В `detect.py`:

```python
detect_single_image(model, image_path, output_dir, conf_threshold=0.3)  # более низкий порог
```

## 🔧 Требования

### Системные требования
- Python 3.8+
- CUDA 11.8+ (для GPU)
- 8GB+ RAM (рекомендуется)
- 2GB+ свободного места

### Зависимости
- ultralytics>=8.0.0
- opencv-python>=4.8.0
- torch>=2.0.0
- onnx>=1.14.0
- onnxruntime>=1.15.0

## 📈 Мониторинг обучения

Во время обучения отслеживайте метрики:
- **mAP50** - средняя точность при IoU=0.5
- **mAP50-95** - средняя точность при IoU=0.5-0.95
- **Precision** - точность
- **Recall** - полнота

Графики сохраняются в `runs/damage/`:
- `results.png` - график метрик
- `confusion_matrix.png` - матрица ошибок
- `val_batch0_pred.jpg` - примеры предсказаний

## 🐛 Решение проблем

### Ошибка "CUDA out of memory"
```python
# Уменьшите размер батча
training_args['batch'] = 8  # или меньше
```

### Ошибка "No module named 'ultralytics'"
```bash
pip install ultralytics
```

### Модель не находит объекты
- Проверьте качество аннотаций
- Уменьшите порог детекции (`conf_threshold`)
- Увеличьте количество эпох обучения

### Медленное обучение на CPU
- Используйте GPU если доступен
- Уменьшите размер изображений (`imgsz`)
- Уменьшите размер батча

## 📝 Лицензия

Этот проект создан в образовательных целях. Используйте на свой страх и риск.

## 🤝 Вклад

Приветствуются улучшения и исправления! Создавайте issues и pull requests.

## 📞 Поддержка

При возникновении проблем:
1. Проверьте версии зависимостей
2. Убедитесь в правильности структуры датасета
3. Проверьте логи обучения
4. Создайте issue с описанием проблемы

---

**Удачного обучения! 🚀**
