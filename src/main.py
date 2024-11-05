import os

from PIL import Image
import streamlit as st
from streamlit import chat_message, text_input
from streamlit_option_menu import option_menu

from gemini_utility import (load_gemini_pro_model,
                            load_gemini_pro_vision_model,
                            embeddings_response,
                            gemini_pro_response,
                            load_environment)


working_directory = os.path.dirname(os.path.abspath(__file__))
load_environment()

st.set_page_config(
    page_title = "Gemini AI",
    page_icon="🧠",
    layout="centered"
)

with st.sidebar:
    selected = option_menu("Gemini AI",
                           ["ChatBot", "Image Captioning", "Embed text", "Ask me anything.."],
                           menu_icon="robot",
                           icons=["chat-dots-fill", "image-fill", "textarea-t", "patch-question-fill"],
                           default_index=0
                           )

def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

if selected == "ChatBot":

    model = load_gemini_pro_model()

    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history = [])

    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    user_prompt = st.chat_input("Ask Gemini Pro...")

    if user_prompt:
        st.chat_message("user").markdown(user_prompt)

        gemini_response = st.session_state.chat_session.send_message(user_prompt)

        with chat_message("assistant"):
            st.markdown(gemini_response.text)

elif selected == "Image Captioning":
    st.title("📷 Snap Captioning")

    uploaded_image = st.file_uploader("Upload an image...", type = ["png", "jpg", "jpeg"])


    #Code for hardcoded prompt of the user

    # if st.button("Generate Caption"):
    #     image = Image.open(uploaded_image)
    #
    #     col1, col2 = st.columns(2)
    #
    #     with col1:
    #         resized_image = image.resize((800,500))
    #         st.image(resized_image)
    #
    #     # default_prompt = "Write a short caption for this image."
    #
    #     caption = load_gemini_pro_vision_model(default_prompt, image)

    user_input = st.chat_input("Ask anything about uploaded image...", )

    if user_input:
        image = Image.open(uploaded_image)

        col1, col2 = st.columns(2)

        with col1:
            resized_image = image.resize((500,300))
            st.image(resized_image)

        # default_prompt = "Write a short caption for this image."

        caption = load_gemini_pro_vision_model(user_input, image)

        with col2:
            st.info(caption)

elif selected=="Embed text":
    st.title("🔠 Text Embed")
    text_input = st.text_area(label="", placeholder="Type the text you want to embed....")
    if st.button("Embed text"):
        response = embeddings_response(text_input)
        st.markdown(response)

elif selected=="Ask me anything..":
    st.title("Ask me ❓")

    user_prompt = st.text_area(label="", placeholder="Ask me anything..")
    if st.button("Get response"):
        response = gemini_pro_response(user_prompt)
        st.markdown(response)
