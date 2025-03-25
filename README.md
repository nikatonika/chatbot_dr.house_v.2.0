# 🧠 Dr. House Chatbot (LLaMA3 fine-tuned)

Интерактивный чат-бот в стиле доктора Хауса, основанный на дообученной модели `LLaMA 3`, с несколькими вариантами инференса и сравнением моделей.

![Chatbot Screenshot](reports/screenshot.png)
---

💬 Пример диалога с моделью:
🧠 I'm Dr. House. Let's get this over with.

👩 Hello doc  
🧠 Hello. So, you're the guy that's been screwing with the team for months, but you're not willing to admit that you're wrong.

👩 I have chest pain. Should I be worried?  
🧠 No. I'm sure you'll be back to your normal self in a few hours. You've got a good heart.

👩 It started after I climbed the stairs. Is it serious?  
🧠 It's a classic symptom of pulmonary embolism. You're in the clear. Just take some aspirin and go back to your normal self.

👩 I also feel short of breath. What could it be?  
🧠 Start her on IV heparin. We'll get her back on track in no time.

👩 My father had a heart attack at 50. Is this genetic?  
🧠 I don't know. It's not the heart. It's the lungs.

👩 Can I take aspirin to help?  
🧠 That's the spirit. Let's get you back on track.

👩 Will you order any tests?  
🧠 You're welcome. I'm going to go get...

👩 What do you think it is?  
🧠 It's a pulmonary embolism. Just like that.

👩 Could it be just anxiety?  
🧠 No. It's a classic symptom of pulmonary embolism.

👩 Should I go to the ER?  
🧠 If you feel any pain, get out of there. You'll be back in two days.

👩 Are you going to help me or just be sarcastic?  
🧠 I'm going to help you. And when you're back on track, you'll be back on track.


## 🧠 Асинхронный Gradio web-сервис

Модель развернута как чат-бот с использованием библиотеки Gradio.  
Асинхронный режим работы реализован через `.queue()`, что позволяет обрабатывать несколько запросов одновременно, не блокируя интерфейс.

Файл запуска: `models_inferences/inference_gradio_async.py`

### Особенности реализации:
- Поддержка `type="messages"` для форматирования в стиле OpenAI
- Контекст сохраняется в `gr.State(...)` как список реплик с ролями
- Ответы Хауса автоматически обрезаются по завершённым 2–3 предложениям (учитываются `.`, `!`, `?`)
- Настройки управления генерацией: `max_tokens`, `temperature`, `top_p`
- Очистка истории и повторные запросы через интерфейс
- Генерация нескольких вариантов ответа и ранжирование по длине и завершенности

Интерфейс запускается автоматически через `demo.queue().launch(share=True)`

✅ Рекомендуется использовать `inference_gradio_async.py` как основную версию web-интерфейса, так как она:
- поддерживает асинхронную очередь
- сохраняет контекст диалога
- лучше управляет генерацией и стилем

## 🤖 Обзор и сравнение обученных моделей доктора Хауса

В ходе проекта было протестировано и дообучено несколько моделей с целью стилизации под персонажа доктора Хауса. Ниже представлены основные результаты и сравнение.

### 🔬 Модель 1: `housemd-chatbot-llama3-lora`
- **Архитектура:** `unsloth/Llama-3.2-1B-Instruct`
- **Формат:** SFT (ручной prompt, без chat_template)
- **Адаптация:** QLoRA (4-bit)
- **Плюсы:**
  - Выдаёт колкие, короткие ответы
  - Хорошо воспроизводит стиль Хауса
  - Устойчиво работает в Colab, быстрая загрузка
- **Минусы:**
  - Иногда создаёт нарративные сценки
  - Повторы и нестабильная структура ответа

### 🔬 Модель 2: `housemd-chatbot-llama3-v3_ext`
- **Архитектура:** `openlm-research/open_llama_3b_v2` (≈3B параметров)
- **Формат:** SFT, anchor → response
- **Адаптация:** LoRA на q_proj/v_proj, без chat_template
- **Плюсы:**
  - Стабильность генерации
  - Выше качество вывода при большей глубине модели
  - Работает с 8-bit квантизацией в Colab
- **Минусы:**
  - Отвечает вопросом на вопрос
  - Потеря стилистики без жёсткого контроля

### 🔬 Модель 3: `housemd-chatbot-llama3-lora_v3_ext_chat_template`
- **Архитектура:** `unsloth/Llama-3.2-1B-Instruct`
- **Формат:** SFT с `chat_template`
- **Адаптация:** QLoRA + LoRA
- **Плюсы:**
  - Чёткий стиль Хауса: сарказм, краткость, логика
  - Обрезка по завершённым предложениям
  - Поддержка диалога через Gradio `type="messages"`
- **Минусы:**
  - Могут встречаться повторы фраз (в рамках образа)
  - Требует чёткой настройки prompt

### 📊 Сравнительная таблица

| Критерий               | llama3-lora_v3_ext_chat_template | llama3-lora               | llama3-v3_ext (3B)       |
|------------------------|-----------------------------------|----------------------------|--------------------------|
| Стиль Хауса            | ✅✅✅                               | ✅✅                        | ✅                       |
| Краткость              | ✅✅✅                               | ✅✅                        | ⚠️ (зависит от вывода)     |
| Логичность             | ✅✅                                | ⚠️                         | ✅✅                     |
| Галлюцинации           | ⚠️                                 | ⚠️                         | ⚠️⚠️                     |
| Скорость генерации     | 5–6 токенов/сек                    | 6–8 токенов/сек            | 3–4 токена/сек           |
| Поддержка chat_template| ✅                                 | ❌                         | ❌                       |

### ✅ Финальный выбор
Для реалистичного чат-бота с лаконичными, резкими и узнаваемыми репликами в духе Грегори Хауса была выбрана модель:

**`housemd-chatbot-llama3-lora_v3_ext_chat_template`**

Она обеспечивает:
- стабильную генерацию;
- фактологическую связность;
- соответствие стилистике персонажа.

Идеально подходит для Gradio-интерфейса с историей чата и асинхронной генерацией ответов.

## 📁 Структура проекта

chatbot_dr.house_v.2.0/
├── assets/                          # Аватары для интерфейса
│   ├── Dr. House.png
│   └── user.jpg
│
├── data/                            # Датасеты
│   ├── questions_answers.csv
│   └── questions_answers_new.csv
│
├── models_inferences/              # Инференсы и web-сервисы
│   ├── inference.ipynb
│   ├── inference_gradio.ipynb
│   ├── inference_gradio_async.py         # Асинхронный Gradio web-сервис
│   ├── inference_gradio_chat_template.ipynb
│   └── inferences_(models_comparison).ipynb  # Сравнение моделей по точности и скорости
│
├── models_training/                # Обучение моделей
│   ├── data_preparing.ipynb
│   ├── train_openlm_research_open_llama_3b_v2_v3_ext.ipynb
│   ├── train_unsloth_Llama_3_2_1B.ipynb
│   └── train_unsloth_Llama_3.2-1B-Instruct_v1_ext_chat_template.ipynb
│
├── reports/                        # Отчёты и иллюстрации
│   ├── Screenshot.png
│   └── Отчет чатбот доктора Хауса 2.0.pdf
│
├── app.py                          # Альтернативный запуск web-интерфейса
├── requirements.txt               # Зависимости
└── README.md

---

## 🏗️ Используемые технологии

- 🤖 **Модели:** `open_llama_3b_v2`, `unsloth/Llama-3.2-1B-Instruct` + `LoRA` + `QLoRA`
- 🧠 **Фреймворки:** `Transformers`, `PEFT`, `trl`, `Gradio`
- 📊 **Данные:** собраны вручную, стилизованы под стиль персонажа Хауса
- ⚙️ **Инференс:** с ручным prompt'ом (в стиле chat_template)
- 📈 **Метрики:** скорость генерации, стиль ответа, контроль повторов и сарказма

---

## 📌 Возможности

- Генерация ответов в стиле Грегори Хауса
- Поддержка нескольких форматов обучения: SFT, chat-style
- Сравнение качества и стиля нескольких моделей
- Web-интерфейс на Gradio с аватарами
- Постобработка и тримминг по второй точке/восклицательному/вопросительному знаку
- Поддержка QLoRA и low-RAM инференса

---

## 🚀 Быстрый старт

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск Gradio-интерфейса
python models_inferences/inference_gradio_async.py
```

---

## 🧪 Инференс моделей

Инференс реализован в двух вариантах:
- `generate_house_response()` — через ручной промпт (в стиле chat-template)
- `setup_chat_format()` — через встроенный `chat_template` HuggingFace (в экспериментальной версии)

Промпты задаются вручную или автоматически, в зависимости от модели.


## 📊 Сравнение моделей
Проведено сравнение 3 подходов:

| Модель                          | Стиль             | Контроль | Скорость | Итог                  |
|--------------------------------|-------------------|----------|----------|------------------------|
| open_llama_3b_v2               | слишком мягкий    | частичный| ✅       | базовая, без стиля     |
| housemd-chatbot-llama3-lora    | резкий и краткий  | полный   | ✅✅      | ближе к Хаусу          |
| housemd-chatbot-llama3-v3_ext  | гладкий и связный | частичный| ✅✅✅    | лучше для чата         |

---

## 📌 Пример генерации

```
📨 User: Can I take painkillers?
🧠 Dr. House: They'll make you feel a little better, but they'll also make you feel a lot worse.
```

---

## 📎 Особенности реализации и соответствие заданию

### ✅ Асинхронный web-сервис
- Модель упакована в Gradio-интерфейс [`inference_gradio_async.py`](models_inferences/inference_gradio_async.py), работающий в асинхронном режиме (`async def`).
- Интерфейс не блокируется при генерации — можно обслуживать несколько пользователей одновременно без задержек.

---

### ✅ Методы повышения фактологической связности

В проекте реализованы базовые методы улучшения логики и качества ответов:

| Метод                   | Описание                                                               | Реализация                                |
|------------------------|------------------------------------------------------------------------|-------------------------------------------|
| 🔍 Постобработка        | Обрезка ответа по завершённым предложениям (., !, ?)                    | `clean_response()`                        |
| 📚 Контекстуальный ввод | Возможность вручную добавлять факты о пациенте в prompt                 | Через ручной prompt в `Gradio`            |
| 🧠 Ранжирование ответов | Генерация нескольких вариантов и выбор лучшего                         | `num_return_sequences + сортировка`      |
| 🤖 Контроль логики      | Возможность второго запроса типа “Does it make sense?” (экспериментально) | Потенциальная доработка                   |

---



## 📄 Дополнительные материалы

- [`inference_gradio_async.py`](models_inferences/inference_gradio_async.py) — финальная реализация инференса в web-интерфейсе с настройками генерации.
- [`Отчет чатбот доктора Хауса 2.0.pdf`](reports/Отчет%20чатбот%20доктора%20Хауса%202.0.pdf) — подробный отчёт с таблицами, графиками, скриншотами и описанием моделей.
- [`inferences_(models_comparison).ipynb`](models_inferences/inferences_(models_comparison).ipynb) — сравнение нескольких версий модели и обоснование финального выбора.
