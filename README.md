
## Установка и запуск

### 1. Создание виртуального окружения

python -m venv venv
source venv/bin/activate


### 2. Установка зависимостей

pip install -r requirements.txt


### 3. Запуск Jupyter Notebook

jupyter notebook attention_visualization.ipynb

## Что включает проект

**Часть 1:**
- Загрузка моделей BERT и GPT-2
- Извлечение attention весов
- Базовые heatmap визуализации
- Сравнение архитектур (bidirectional vs causal)
- Анализ attention в разных слоях

**Часть 2:**
- Multi-Head Attention анализ
- Загрузка русскоязычной модели (RuBERT)
- Кросс-языковое исследование
- Практические выводы
