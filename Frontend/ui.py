import gradio as gr
import requests

API_URL = "http://127.0.0.1:5000/predict"

def classify_image(image_path):
    try:
        with open(image_path, "rb") as f:
            files = {"image": f}
            response = requests.post(API_URL, files=files, timeout=10) 
        
        if response.status_code != 200:
            return f"❌ Lỗi Server: {response.text}"

        result = response.json()
        
        label = result.get('label', 'Không xác định')
        conf = result.get('confidence', 0)
        
        return f"🍀 Dự đoán: {label}\n🎯 Độ tin cậy: {conf*100:.2f}%"

    except requests.exceptions.ConnectionError:
        return "❌ Lỗi: Không thể kết nối tới Backend (Hãy chạy app.py trước!)"
    except Exception as e:
        return f"❌ Lỗi không xác định: {str(e)}"

interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="filepath", label="Upload ảnh trái cây / rau củ"),
    outputs=gr.Textbox(label="Kết quả nhận diện"),
    title="🥦 AI Nhận diện Trái cây & Rau củ",
    description="Quy trình: Image -> Gradio -> Flask API -> Keras Model"
)

if __name__ == "__main__":
    interface.launch(share=False, server_port=7861)
