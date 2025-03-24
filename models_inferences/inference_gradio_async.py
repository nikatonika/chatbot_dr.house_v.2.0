# app.py

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM

# === Путь к модели с chat_template ===
model_path = "nikatonika/Llama-3.2-1B-Instruct_v1_ext_chat_template"

# === Загрузка модели и токенизатора ===
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

model = AutoPeftModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# === Генерация ===
async def generate_response(user_input, max_new_tokens, temperature, top_p):
    # Пример внешнего контекста (можно заменить на реальные данные пациента)
    context = "Patient is female, 43, reports fatigue and high stress."

    prompt = f"""{context}

You are Dr. Gregory House, a world-class diagnostician known for sarcasm, wit, and medical expertise.
You don't sugarcoat anything and always rely on logic and medical facts.

Answer concisely, with dry humor and intelligence.

User: {user_input}
Dr. House:"""

    inputs = tokenizer(prompt.strip(), return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=3,  # Генерация нескольких вариантов
            pad_token_id=tokenizer.eos_token_id
        )

    responses = [tokenizer.decode(o, skip_special_tokens=True).split("Dr. House:")[-1].strip() for o in outputs]

    # Простой ранжировщик: выбрать самый короткий адекватный ответ, содержащий хотя бы одну точку
    best = sorted(responses, key=lambda r: (len(r.split()), -r.count(".")))[0]

    # Обрезка по второй точке
    if best.count(".") >= 2:
        best = ".".join(best.split(".")[:2]) + "."

    return best

# === Gradio-интерфейс ===
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 HouseMD Chatbot — Llama-3.2-1B ChatTemplate")
    gr.Markdown("Дообученная модель, имитирующая стиль доктора Хауса. Используется `chat_template` и ручной prompt с ранжированием.")

    with gr.Row():
        with gr.Column(scale=2):
            user_input = gr.Textbox(label="Вопрос пациентки", placeholder="Why am I still sick?")
            max_tokens = gr.Slider(32, 200, value=80, step=8, label="Максимум токенов")
            temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
            top_p = gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="Top-p")
            button = gr.Button("Спросить доктора Хауса")
        with gr.Column(scale=3):
            output = gr.Textbox(label="Ответ Хауса", lines=4)

    button.click(fn=generate_response,
                 inputs=[user_input, max_tokens, temperature, top_p],
                 outputs=output)

if __name__ == "__main__":
    demo.launch()