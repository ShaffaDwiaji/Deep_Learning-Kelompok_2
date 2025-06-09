import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Load model function
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load("model_sapi.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Image transform
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def main():
    st.title("Deteksi Sapi Sakit atau Sehat 🐄")
    st.write("Upload gambar sapi untuk mendeteksi kesehatannya.")

    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Gambar yang diupload', use_container_width=True)

        model = load_model()
        input_tensor = transform_image(image)
        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)

        label = "Sakit" if pred.item() == 0 else "Sehat"
        st.subheader(f"Hasil Deteksi: **{label}**")

if __name__ == "__main__":
    main()
