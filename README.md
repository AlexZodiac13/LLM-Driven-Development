# Оценка качества генерации текста (Task 4)

Проект выполнен в рамках домашнего задания по курсу LLM-Driven Development.
Выбран Трек B: Text Summarization (Суммаризация текстов).

## Структура проекта
- `solution.ipynb`: Основной Jupyter Notebook с кодом экспериментов.
- `requirements.txt`: Список зависимостей.
- `.venv`: Виртуальное окружение (должно быть создано).

## Установка и запуск

1. Перейдите в папку проекта:
   ```bash
   cd task4_project
   ```

2. (Если не создано) Создайте виртуальное окружение:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

4. Запустите Jupyter Notebook:
   ```bash
   jupyter notebook solution.ipynb
   ```
   Или откройте файл в VS Code и выберите ядро `.venv`.

## Выбор моделей и данных
- **Датасет**: `IlyaGusev/gazeta` (Тестовая выборка).
- **Модели**:
  1. `IlyaGusev/rut5_base_sum_gazeta` (Специализированная T5).
  2. `cointegrated/rut5-base-multitask` (Универсальная T5).

## Метрики
Реализован подсчет:
- ROUGE-1, ROUGE-L
- BLEU
- BERTScore (Semantic Similarity)
- Время генерации и длина ответа.
