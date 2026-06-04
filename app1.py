import streamlit as st

st.set_page_config(
    'Cats and Dogs',
    "🐱&🐶",
    "centered",
    'auto'
)

st.title('Cats VS Dogs')
st.header('This is an simple tool which can be used to recognize cats and dogs!')
st.subheader("A small project developed by Bogger!")

st.sidebar.header("Setting:")
model_choice =  st.sidebar.selectbox("Chose the model you want to use.",[
    'CNN','ResNet18','VIT'
])
st.sidebar.write(f"You select {model_choice}.")
lr = st.sidebar.number_input("Chose the learning rate of you test!",min_value=-4,max_value=1,value=-2,)
epoches = st.sidebar.slider("Setting the epoches of you training!",min_value=1,max_value=100,value=10)


iuput_image = st.file_uploader("Upload the target image!",type=['jpg','png'])

if st.button('Start to recognize!'):
    st.text("Running~")