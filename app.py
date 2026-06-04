import streamlit as st
from PIL import Image
import torch
from torchvision import transforms

st.set_page_config(
    "Dogs and Cats",
    None,
    "centered",
    'auto'
)

st.title('Dogs and Cats')
st.header("Input an image and we can tell whether is a cat or a dog!")
st.subheader("Based on ResNet18 and PyTorch")
st.write("by bogger")
st.markdown("**jia cu**,*xieti*,'daimakuai',#biaoti")
st.text("Hello World!")
st.code("print('Hello World')",language='python')
st.markdown("MarkDown Text")

img = Image.open("TestFile\GPT.png")

st.image(img,caption='example',width='content')

st.video(r'TestFile\nige.mp4',loop=True,muted=True,width='stretch')
st.audio(r'TestFile\nige.mp3')

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,10,100)
y = np.sin(x)
fig, ax = plt.subplots()
ax.plot(x,y)
st.pyplot(fig)

fig, ax = plt.subplots()
ax.plot(x, np.sin(x), label='sin(x)')
ax.plot(x, np.cos(x), label='cos(x)')
ax.plot(x, np.sin(2*x), label='sin(2x)')
ax.legend()
st.pyplot(fig)

fig, axes = plt.subplots(3, 1, figsize=(6, 8)) 
axes[0].plot(x, np.sin(x))
axes[0].set_title('sin(x)')

axes[1].plot(x, np.cos(x))
axes[1].set_title('cos(x)')

axes[2].plot(x, np.sin(2*x))
axes[2].set_title('sin(2x)')

st.pyplot(fig)

uploaded_file = st.file_uploader("上传图片", type=["jpg","png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)

if st.button("点击预测"):
    st.write("按钮被点击了")

threshold = st.slider("置信度阈值", min_value=0.0, max_value=1.0, value=0.5)
num_epochs = st.number_input("训练轮数", min_value=1, max_value=100, value=10)

option = st.selectbox(
    "选择类别",
    ["猫", "狗", "其他"]
)
st.write("你选择了:", option)

multi_options = st.multiselect(
    "选择多个选项",
    ["猫", "狗", "兔子"]
)

# 列布局
col1, col2 = st.columns(2)
col1.header("左边列")
col1.image("TestFile\GPT.png")
col2.header("右边列")
col2.image("TestFile\GPT.png")

# 侧边栏
st.sidebar.title("设置")
confidence = st.sidebar.slider("置信度阈值", 0.0, 1.0, 0.5)