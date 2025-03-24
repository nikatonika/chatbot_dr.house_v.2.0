# 🧠 Dr. House Chatbot (LLaMA3 fine-tuned)

Интерактивный чат-бот в стиле доктора Хауса, основанный на дообученной модели `LLaMA 3`, с несколькими вариантами инференса и сравнением моделей.

---

## 📁 Структура проекта

chatbot_dr.house_v.2.0/ 
chatbot_dr.house_v.2.0/
├── assets/                        # Аватары для интерфейса
│   ├── Dr. House.png
│   └── user.jpg
│
├── data/                          # Датасеты
│   ├── questions_answers.csv
│   └── questions_answers_new.csv
│
├── models_inferences/            # Инференсы и сравнение
│   ├── inference.ipynb
│   ├── inference_gradio.ipynb
│   ├── inference_gradio_async.py   # Асинхронный Gradio web-сервис
│   └── inferences_(models_comparison).ipynb
│
├── models_training/              # Обучение моделей
│   ├── data_preparing.ipynb
│   ├── train_openlm_research_open_llama_3b_v2.ipynb
│   ├── train_unsloth_Llama_3_2_1B.ipynb
│   └── train_unsloth_Llama-3.2-1B-lora_chat_template.ipynb
│
├── reports/
│   └── Отчет чатбот доктора Хауса 2.0.pdf
│
├── app.py                        # Альтернативный запуск web-интерфейса
├── requirements.txt             # Зависимости проекта
└── README.md                    # Этот файл


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


---

## 📎 Особенности реализации и соответствие заданию

### ✅ Асинхронный web-сервис
- Модель упакована в Gradio-интерфейс [`inference_gradio_async.py`](models_inferences/inference_gradio_async.py), работающий в асинхронном режиме (`async def`).
- Интерфейс не блокируется при генерации — можно обслуживать несколько пользователей одновременно без задержек.

---

### ✅ Методы повышения фактологической связности

В проекте реализованы базовые методы улучшения логики и качества ответов:

| Метод                           | Описание                                                                 | Реализация                                  |
|--------------------------------|--------------------------------------------------------------------------|---------------------------------------------|
| 🔍 Постобработка                | Обрезка ответа по второй точке для завершённой мысли                     | `generate_house_response()`                |
| 📚 Контекстуальный ввод         | Возможность вручную добавлять факты о пациенте в prompt                  | Через ручной prompt в `Gradio`             |
| 🧠 Ранжирование ответов         | Генерация нескольких вариантов и выбор лучшего (опционально)             | Возможность задать `num_return_sequences`  |
| 🤖 Контроль логики              | Возможность второго запроса типа “Does it make sense?” (экспериментально) | Потенциальная доработка                    |

---

### 📄 Дополнительные материалы

- [`inference_gradio_async.py`](models_inferences/inference_gradio_async.py) — финальная реализация инференса в web-интерфейсе с настройками генерации.
- [`Отчет чатбот доктора Хауса 2.0.pdf`](reports/Отчет%20чатбот%20доктора%20Хауса%202.0.pdf) — подробный отчёт с таблицами, графиками, скриншотами и описанием моделей.
- [`inferences_(models_comparison).ipynb`](models_inferences/inferences_(models_comparison).ipynb) — сравнение нескольких версий модели и обоснование финального выбора.
