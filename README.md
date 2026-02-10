# Task 8 — Track C (Medical dialogues): Entity & event extraction

Цель: извлечь структурированные сущности из медицинских диалогов (SYMPTOM, DIAGNOSIS, MEDICATION, DOSAGE, DURATION, SIDE_EFFECT) с помощью локального inference LLM, сравнить скорость/ресурсы и показать базовую оценку качества.

## Что внутри

- `notebooks/task8_track_c.ipynb` — основной ноутбук с комментариями и примерами
- `requirements.txt` — зависимости для виртуального окружения

## Быстрый старт (venv)

```bash
cd task8_track_c
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Запуск Jupyter:

```bash
python -m pip install jupyter
jupyter lab
```

## Примечания по моделям

В ноутбуке предусмотрены 2–3 модели (ID можно менять):

- небольшой instruct-модель для CPU/GPU (по умолчанию)
- альтернативная instruct-модель
- опционально: более тяжёлая медицинская модель (если доступна на машине)

Сравнение `full precision` vs `4-bit` делается для одной и той же модели, если:

- доступна CUDA
- установлен `bitsandbytes`

Если GPU нет — ноутбук всё равно отработает на CPU, но блок про 4-bit будет автоматически пропущен.
