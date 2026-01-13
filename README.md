# Задача 3: Трекинг LLM-инференса с vLLM и MLflow

### 1. Подготовка окружения

1.  Создайте и активируйте виртуальное окружение (если еще не создано):
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  Установите необходимые библиотеки:
    ```bash
    pip install -r requirements.txt
    ```
### 2. Запуск vLLM сервера

Запустите сервер vLLM, который будет имитировать OpenAI API.

```bash
./start_vllm.sh
```

Дождитесь сообщения `Uvicorn running on http://0.0.0.0:8000`. Не закрывайте этот терминал.

### 3. Взаимодействие с моделью

В **новом терминале**  выполните скрипты:

**А. HTTP запрос (httpx):**
```bash
 .venv/bin/python interact_http.py
```

**Б. OpenAI SDK:**
```bash
.venv/bin/python interact_openai.py
```

### 4. Оценка с MLflow (LLM-as-a-Judge)

Теперь запустим эксперимент, где vLLM будет выступать в роли "судьи" и оценивать ответы (метрика Correctness).

1.  (Опционально) Запустите MLflow UI в отдельном терминале:
    ```bash
     .venv/bin/python -m mlflow ui --host 0.0.0.0 --port 5001

    ```

2.  Запустите скрипт оценки:
    ```bash
    .venv/bin/python mlflow_eval.py
    ```

