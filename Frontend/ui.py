import gradio as gr
import requests

API_URL = "http://127.0.0.1:5000/predict"

def classify_image(image):
    response = requests.post(
        API_URL,
        files={"image": open(image, "rb")}
    )
    result = response.json()
    return f"🍀 Dự đoán: {result['label']}\n🎯 Độ tin cậy: {result['confidence']*100:.2f}%"

interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="filepath", label="Upload ảnh trái cây / rau củ"),
    outputs=gr.Textbox(label="Kết quả nhận diện"),
    title="🥦 AI Nhận diện Trái cây & Rau củ",
    description="Sử dụng Teachable Machine + Flask + Gradio"
)

interface.launch()
