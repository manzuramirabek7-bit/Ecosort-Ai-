import os
import json
import asyncio
import numpy as np
import gradio as gr
import edge_tts
from PIL import Image
from tensorflow import keras


import os
import gdown

MODEL_PATH = "ecosort_model.keras"
LABELS_PATH = "class_names.json"
if not os.path.exists(MODEL_PATH):
    gdown.download(
        id="1POdnaRmf4Cn7V86kh3iQXtIi8nE_zU5o",
        output=MODEL_PATH,
        quiet=False
    )

model = keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

kazakh_names = {
    "GLASS": "ШЫНЫ",
    "METAL": "МЕТАЛЛ",
    "PAPER": "ҚАҒАЗ",
    "PLASTIC": "ПЛАСТИК"
}

voice_messages = {
    "GLASS": "Шыны санаты анықталды. Шыныны шыныға арналған жәшікке салыңыз!",
    "METAL": "Металл санаты анықталды. Металды металлға арналған жәшікке салыңыз!",
    "PAPER": "Қағаз санаты анықталды. Қағазды қағазға арналған жәшікке салыңыз!",
    "PLASTIC": "Пластик санаты анықталды. Пластикті пластикке арналған жәшікке салыңыз!"
}

async def create_voice(text, path):
    communicator = edge_tts.Communicate(
        text=text, voice="kk-KZ-AigulNeural"
    )
    await communicator.save(path)

def predict_waste(image):
    if image is None:
        return {"Сурет жүктелмеді": 1.0}, "Алдымен сурет жүктеңіз.", None

    image = image.convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(image, dtype=np.float32), axis=0)

    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])

    results = {
        kazakh_names[name]: float(probs[i])
        for i, name in enumerate(class_names)
    }

    if confidence < 0.60:
        return results, "⚠️ Сенімділік төмен. Басқа сурет қолданып көріңіз.", None

    category = class_names[idx]
    message = (
        f"♻️ Анықталған қалдық: {kazakh_names[category]}\n"
        f"Болжам сенімділігі: {confidence * 100:.1f}%"
    )

    audio_path = f"/tmp/ecosort_{category.lower()}.mp3"
    try:
        asyncio.run(create_voice(voice_messages[category], audio_path))
    except Exception as e:
        print("Дыбыс қатесі:", e)
        audio_path = None

    return results, message, audio_path

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # ♻️ EcoSort AI
    ### Камера көрді — AI сұрыптады!
    Қалдықтың суретін жүктеңіз немесе камерамен түсіріңіз.
    """)

    with gr.Row():
        image_input = gr.Image(
            label="Қалдықтың суреті",
            type="pil",
            sources=["upload", "webcam"]
        )
        prediction_output = gr.Label(
            label="Санаттар бойынша болжам",
            num_top_classes=4
        )

    result_text = gr.Textbox(label="Жүйе қорытындысы", lines=3)
    voice_output = gr.Audio(
        label="🔊 Қазақша дыбыстық нәтиже",
        type="filepath",
        autoplay=True
    )

    button = gr.Button("♻️ Қалдықты анықтау", variant="primary")
    button.click(
        fn=predict_waste,
        inputs=[image_input],
        outputs=[prediction_output, result_text, voice_output]
    )

demo.launch()
