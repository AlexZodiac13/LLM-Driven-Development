## 1) Установка Python 3.12

**macOS (Homebrew)**

```bash
brew install python@3.12
echo 'export PATH="/opt/homebrew/opt/python@3.12/libexec/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc
python3.12 --version
```

**Windows (PowerShell)**

```powershell
winget install -e --id Python.Python.3.12
python --version
```

**Ubuntu/Debian**

```bash
sudo add-apt-repository ppa:deadsnakes/ppa -y && sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev
python3.12 --version
```

---

## 2) Создание виртуального окружения и установка зависимостей

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .\llm-venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**requirements.txt (фиксированные версии)**

```txt
ragas==0.3.4
pandas==2.3.2
python-dotenv==1.1.1
tqdm==4.67.1
rich==14.1.0
openai==1.107.3
langchain-openai==0.3.33
datasets==4.0.0
pyarrow==17.0.0
```

---

## 3) Запуск тестов Ragas

### Обычный запуск (Python)

Создайте файл `.env`. Пример:
```ini
LLM_PROVIDER=local
# Если использовать Yandex GPT:
# LLM_PROVIDER=yandex
# YC_API_KEY=...
# YC_FOLDER_ID=...
```

Запустите скрипт:
```bash
python run_ragas_demo_test.py --goldens tests/goldens.json --output tests/results.json
```

---

## Метрики и Пороговые значения (Thresholds)

Для оценки RAG-пайплайна используются следующие метрики. Пороговые значения автоматически адаптируются в зависимости от переменной окружения `LLM_PROVIDER` ("local" или "yandex/openai").

| Метрика | Описание | Порог (Prod) | Порог (Local) |
|:---|:---|:---:|:---:|
| **Faithfulness** | Оценивает верность контексту (проверка на галлюцинации). | 0.80 | 0.01 |
| **Answer Relevancy** | Оценивает релевантность ответа заданному вопросу. | 0.50 | 0.10 |
| **Context Precision** | Проверяет, насколько высоко ранжирован релевантный контекст. | 0.50 | 0.01 |
| **Context Recall** | Проверяет, соответствует ли найденный контекст эталону (Ground Truth). | 0.70 | 0.01 |
| **QA Semantic Correctness** | Кастомная метрика: косинусное сходство между Ответом и Эталоном. | 0.80 | 0.60 |

*Примечание: Пороги для локальной разработки снижены из-за ограничений маленьких моделей (например, Llama 3.2 3B), которые могут некорректно форматировать выходные данные для метрик Ragas/JSON.*

---

## Пайплайн CI/CD

Проект поддерживает полностью локальный CI/CD пайплайн на базе Docker Compose.

### Локальный запуск (Docker)

Чтобы запустить полный набор тестов в чистом, изолированном окружении:

```bash
docker compose up --build --abort-on-container-exit
```

Эта команда:
1. Запускает контейнер `ollama`.
2. Загружает модель `llama3.2`.
3. Запускает скрипт оценки Ragas внутри контейнера `ragas-tests`.
4. Монтирует папку `./tests/` для сохранения отчета `test_results.json`.
5. Завершается с кодом 0, если тесты прошли (метрики > порогов), или с кодом 1, если упали.

### GitHub Actions / GitLab CI

Для интеграции в CI-провайдер достаточно выполнить ту же команду Docker. Пример для GitHub Actions:

```yaml
jobs:
  ragas-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Ragas Tests
        run: docker compose up --build --abort-on-container-exit
```
