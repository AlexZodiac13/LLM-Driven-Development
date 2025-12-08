# Визуализация механизма работы Attention

## Установка и запуск

### 1. Создание виртуального окружения (рекомендуется)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 3. Запуск Jupyter Notebook

```bash
jupyter notebook attention_visualization.ipynb
```

## Структура проекта

```
d:\mygit\LLM-Driven-Development\
├── requirements.txt              # Зависимости
├── README.md                     # Этот файл
├── attention_visualization.ipynb # Основной ноутбук с заданиями
└── utils/
    ├── __init__.py
    ├── attention_extractor.py    # Утилиты для извлечения attention
    └── visualization.py          # Функции визуализации
```

## Что включает проект

✅ **Часть 1 (70% оценки):**
- Загрузка моделей BERT и GPT-2
- Извлечение attention весов
- Базовые heatmap визуализации
- Сравнение архитектур (bidirectional vs causal)
- Анализ attention в разных слоях

✅ **Часть 2 (30% оценки):**
- Multi-Head Attention анализ
- Загрузка русскоязычной модели (RuBERT)
- Кросс-языковое исследование
- Практические выводы

## Требования к системе

- Python 3.8+
- 4GB+ RAM (рекомендуется 8GB для загрузки моделей)
- GPU опционально (ускорит обработку)

## Временные затраты

- Первый запуск: ~10-15 минут (загрузка моделей)
- Последующие запуски: ~2-3 минуты
