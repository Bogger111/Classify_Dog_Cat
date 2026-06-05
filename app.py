from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from CNNnet import MyCNN
from Resnet import MyResNet18


BASE_DIR = Path(__file__).resolve().parent
CLASS_NAMES = ["猫", "狗"]
MODEL_OPTIONS = {
    "ResNet18": {
        "class": MyResNet18,
        "checkpoint": BASE_DIR / "checkpoints" / "Adam_lr0.001_CrossEntropyLoss_StepLR_best.pth",
    },
    "CNN": {
        "class": MyCNN,
        "checkpoint": BASE_DIR / "checkpoints" / "CNN_best.pth",
    },
}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model(model_name):
    device = get_device()
    model_info = MODEL_OPTIONS[model_name]
    model = model_info["class"](num_classes=2, dropout=0.3)

    checkpoint = torch.load(model_info["checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model


def preprocess_image(image):
    transform = transforms.Compose(
        [
            transforms.Resize(260),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transform(image.convert("RGB")).unsqueeze(0)


def predict(image, model):
    device = get_device()
    image_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, predicted_index = torch.max(probabilities, dim=0)

    return CLASS_NAMES[predicted_index.item()], confidence.item()


st.set_page_config(page_title="猫狗识别", page_icon="🐱", layout="centered")

st.title("猫狗识别")
st.write("上传一张图片，选择模型后进行识别。")

model_name = st.sidebar.selectbox("选择模型", list(MODEL_OPTIONS.keys()))
image_width = st.sidebar.slider("图片显示宽度", 200, 800, 420, 20)
checkpoint_path = MODEL_OPTIONS[model_name]["checkpoint"]

if not checkpoint_path.exists():
    st.sidebar.error(f"{model_name} 权重文件不存在")
    st.error(f"没有找到模型文件：{checkpoint_path}")
    if model_name == "CNN":
        st.info("请先训练 CNN，并把最佳权重保存为 checkpoints/CNN_best.pth。")
    st.stop()

uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="待识别图片", width=image_width)

    if st.button("开始识别", type="primary"):
        with st.spinner("正在识别..."):
            model = load_model(model_name)
            label, confidence = predict(image, model)

        st.success(f"预测结果：{label}")
        st.metric("置信度", f"{confidence:.2%}")
