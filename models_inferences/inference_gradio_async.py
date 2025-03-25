# inference_gradio_async.py

import torch
import gradio as gr
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import re

# === Путь к модели с chat_template ===
model_path = "nikatonika/Llama-3.2-1B-Instruct_v1_ext_chat_template"

# === Загрузка модели и токенизатора ===
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoPeftModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# === Очистка и обрезка до 2–3 предложений
def clean_response(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return " ".join(sentences[:3]) if len(sentences) >= 3 else text.strip()

# === Асинхронная генерация
async def generate_house_response(user_input, history, max_new_tokens=80, temperature=0.7, top_p=0.9):
    if history is None or not isinstance(history, list):
        history = []

    history.append({"role": "user", "content": user_input})

    # Внешний контекст (можно адаптировать)
    context = "Patient is female, 43, reports fatigue and stress."

    # Формирование chat-like prompt
    prompt = f"{context}\n\nYou are Dr. Gregory House, a world-class diagnostician known for sarcasm, wit, and medical expertise.\n"
    prompt += "You don't sugarcoat anything and always rely on logic and medical facts.\n\n"
    prompt += "Answer concisely, with dry humor and intelligence.\n\n"

    for msg in history:
        if msg["role"] == "user":
            prompt += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            prompt += f"Dr. House: {msg['content']}\n"
    prompt += "Dr. House:"

    inputs = tokenizer(prompt.strip(), return_tensors="pt", truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=3,
            pad_token_id=tokenizer.eos_token_id
        )

    candidates = [tokenizer.decode(o, skip_special_tokens=True).split("Dr. House:")[-1].strip() for o in outputs]
    best = sorted(candidates, key=lambda r: (len(r.split()), -r.count(".")))[0]
    best = clean_response(best)

    history.append({"role": "assistant", "content": best})
    return history, history

# === Gradio-интерфейс
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 HouseMD Chatbot — Llama-3.2-1B (ChatTemplate, Async)")
    gr.Markdown("💬 *Do you really want the truth? Then ask Dr. House.*")

    chatbot = gr.Chatbot(label="🩺 Chat with Dr. House", type="messages")
    state = gr.State([{"role": "assistant", "content": "I'm Dr. House. Let's get this over with."}])

    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Ask your question...")
    with gr.Row():
        max_tokens = gr.Slider(32, 200, value=80, step=8, label="Max Tokens")
        temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
        top_p = gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="Top-p")

    txt.submit(generate_house_response, [txt, state, max_tokens, temperature, top_p], [chatbot, state])
    gr.ClearButton([txt, chatbot, state], value="🧹 Clear")

demo.queue().launch(share=True, debug=True)
