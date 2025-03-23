# 🧠 Dr. House Chatbot (LLaMA3 fine-tuned)

Интерактивный чат-бот в стиле доктора Хауса, основанный на дообученной модели `LLaMA 3`, с несколькими вариантами инференса и сравнением моделей.

---

## 📁 Структура проекта

CHATBOT_DR.HOUSE_V... ├── assets/ # Аватары для чата (пользователь и доктор Хаус) │ ├── Dr. House.png │ └── user.jpg │ ├── data/ # CSV-файлы с вопросами и ответами │ ├── questions_answers.csv # Исходный датасет │ └── questions_answers_new.csv # Обновлённая версия (после доработки) │ ├── models_inferences/ # Инференс и сравнение моделей │ ├── inference_gradio.ipynb # Интерфейс с Gradio │ └── inferences_(models_comparison).ipynb # Сравнение моделей и генерация ответов │ ├── models_training/ # Скрипты для обучения │ ├── data_preparing.ipynb # Подготовка данных │ ├── train_openlm_research_open_llama_3b_v2.ipynb # Дообучение open_llama │ ├── train_unsloth_Llama_3_2_1B.ipynb # Обучение модели unsloth (LoRA + SFT) │ └── train_unsloth_Llama-3.2-1B-lora_chat_template.ipynb # Обучение с chat_template │ ├── reports/ # (Папка для финальных отчётов и метрик) │ ├── venv/ # Виртуальное окружение (в .gitignore) │ ├── requirements.txt # Список зависимостей └── README.md # Этот файл

---

## **Project Structure**
chatbot_dr.house_v.2.0/ 
│── 📂 data/ # Dataset storage │ 
│   ├── questions_answers.csv # Raw dataset (House's quotes and responses)
│   ├── questions_answers.npy # Processed dataset (for training)
│
│── 📂 models/                # Model training
│   ├── train.ipynb           # Notebook for LLM fine-tuning with LoRA
│   ├── config.yaml           # Configuration file for training
│   ├── tokenizer.json        # Tokenizer file
│
│── 📂 api/                   # Web service (FastAPI)
│   ├── app.py               # FastAPI server implementat
│   ├── requirements.txt      # Python dependencies
│   ├── Dockerfile            # Containerization for Google Cloud Run
│
│── 📂 deploy/                # Deployment scripts
│   ├── deploy.sh             # Automated deployment script
│   ├── cloudbuild.yaml       # Google Cloud Build configuration
│
│── README.md                 # Project documentation
│── .gitignore                # Ignored files (models, logs, cache)
---


---

## 🏗️ Используемые технологии

- 🤖 **Модели:** `open_llama_3b_v2`, `unsloth/Llama-3.2-1B-Instruct` + `LoRA` + `QLoRA`
- 🧠 **Фреймворки:** `Transformers`, `PEFT`, `trl`, `Gradio`
- 📊 **Данные:** собраны вручную, стилизованы под стиль персонажа Хауса
- ⚙️ **Инференс:** с кастомным prompt'ом или `chat_template`
- 📈 **Метрики:** скорость генерации, стиль ответа, контроль повторов и сарказма

---

## 📌 Возможности

- Генерация ответов в стиле Грегори Хауса
- Поддержка нескольких форматов обучения: SFT, chat-template
- Сравнение качества и стиля нескольких моделей
- Web-интерфейс на Gradio с аватарами
- Постобработка и тримминг по второй точке
- Поддержка QLoRA и low-RAM инференса

---

## 🚀 Быстрый старт

```bash
# Установка зависимостей
pip install -r requirements.txt


# Запуск Gradio-интерфейса
python models_inferences/inference_gradio.ipynb
🧪 Инференс моделей
Инференс реализован в двух вариантах:

generate_house_response() — через кастомный промпт

setup_chat_format() — через chat_template HuggingFace

Промпты задаются вручную или автоматически, в зависимости от модели.

📊 Сравнение моделей
Проведено сравнение 3 подходов:

Модель	Стиль	Контроль	Скорость	Итог
open_llama_3b_v2	слишком мягкий	частичный	✅	базовая, без стиля
housemd-chatbot-llama3-lora	резкий и краткий	полный	✅✅	ближе к Хаусу
housemd-chatbot-llama3-v3_ext_chat	гладкий и связный	частичный	✅✅✅	лучше для чата
📌 Пример генерации
text
Copy
Edit
📨 User: Can I take painkillers?
🧠 Dr. House: They'll make you feel a little better, but they'll also make you feel a lot worse.
