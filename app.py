import streamlit as st
import torch
from PIL import Image

from src.inference import load_artifacts, generate_caption

st.set_page_config(page_title="Image Captioning", page_icon="🖼️", layout="centered")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def get_model():
    return load_artifacts(device=DEVICE)


st.title("🖼️ Image Captioning")
st.caption("ResNet18 + LSTM + GloVe · генерация текстового описания по изображению")

with st.sidebar:
    st.header("⚙️ Параметры генерации")
    strategy = st.radio("Стратегия сэмплирования", ["greedy", "top-k", "top-p"])
    top_k = top_p = None
    if strategy == "greedy":
        top_k = 1
    elif strategy == "top-k":
        top_k = st.slider("top-k", 2, 20, 5)
    else:
        top_p = st.slider("top-p", 0.1, 1.0, 0.9, 0.05)
    temperature = st.slider("temperature", 0.5, 2.0, 1.0, 0.1)
    n_captions = st.slider("Число вариантов", 1, 5, 3)
    st.divider()
    st.caption(f"Device: `{DEVICE}`")

uploaded = st.file_uploader("Загрузите изображение", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns([1, 1])

if uploaded is not None:
    image = Image.open(uploaded)
    with col1:
        st.image(image, caption="Ваше изображение", use_container_width=True)

    if st.button("✨ Сгенерировать подпись", type="primary", use_container_width=True):
        model, vocab = get_model()
        with col2:
            with st.spinner("Генерируем..."):
                for i in range(n_captions):
                    caption = generate_caption(
                        model, vocab, image, device=DEVICE,
                        top_k=top_k, top_p=top_p, temperature=temperature,
                    )
                    st.success(f"**#{i+1}:** {caption}")
else:
    st.info("👆 Загрузите изображение, чтобы начать")

st.divider()
st.markdown(
    "Сделано в рамках учебного проекта. "
    "[Код на GitHub](https://github.com/demter/image-captioning)"
)
