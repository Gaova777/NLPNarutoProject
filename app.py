import streamlit as st
import pandas as pd
import plotly.express as px
from theme_classifier import ThemeClassifier
from character_network import NamedEntityRecognizer, CharacterNetworkGenerator
from text_classification import JutsuClassifier
import os
from dotenv import load_dotenv

load_dotenv()

# Configuración de la página
st.set_page_config(page_title="Análisis de Series/Películas", layout="wide")

st.title("📺 Análisis de Temas, Redes de Personajes y Clasificación de Texto")

# =======================
# 📌 Funciones de Procesamiento
# =======================

def get_themes(theme_list, subtitles_path, save_path):
    theme_list = theme_list.split(",")
    theme_classifier = ThemeClassifier(theme_list)
    output_df = theme_classifier.get_theme(subtitles_path, save_path)
    theme_list = [theme for theme in theme_list if theme != "dialogue"]
    output_df = output_df[theme_list]
    output_df = output_df.sum().reset_index()
    output_df.columns = ["Theme", "Score"]
    return output_df

def get_character_network(subtitles_path, ner_path):
    ner = NamedEntityRecognizer()
    ner_df = ner.get_ners(subtitles_path, ner_path)
    character_network_generator = CharacterNetworkGenerator()
    relationship_df = character_network_generator.generate_character_network(ner_df)
    html = character_network_generator.draw_network_graph(relationship_df)
    return html

def classify_text(text_classification_model, text_classification_data_path, text_to_classify):
    jutsu_classifier = JutsuClassifier(
        model_path=text_classification_model,
        data_path=text_classification_data_path,
        huggingface_token=os.getenv("huggingface_token")
    )
    output = jutsu_classifier.classify_jutsu(text_to_classify)
    return output

# =======================
# 📌 Sección 1: Clasificación de Temas
# =======================

st.header("🎭 Clasificación de Temas en el Guion")
col1, col2 = st.columns([1, 2])

with col1:
    theme_list = st.text_input("Lista de Temas (separados por coma)", "action,comedy,drama")
    subtitles_path = st.text_input("Ruta de los Subtítulos o Guion", "data/subtitles.csv")
    save_path = st.text_input("Ruta para Guardar el Resultado", "output/themes.csv")

    if st.button("🔍 Obtener Temas"):
        with st.spinner("Procesando..."):
            output_df = get_themes(theme_list, subtitles_path, save_path)
            st.session_state["themes"] = output_df

with col2:
    if "themes" in st.session_state:
        fig = px.bar(
            st.session_state["themes"], x="Theme", y="Score", title="Distribución de Temas",
            color="Theme", text="Score"
        )
        st.plotly_chart(fig, use_container_width=True)

# =======================
# 📌 Sección 2: Red de Personajes
# =======================

st.header("🕵️‍♂️ Red de Personajes en el Guion")
col3, col4 = st.columns([1, 2])

with col3:
    subtitles_path_ner = st.text_input("Ruta de los Subtítulos", "data/subtitles.csv")
    ner_path = st.text_input("Ruta de NER", "output/ners.csv")

    if st.button("🕸️ Generar Red de Personajes"):
        with st.spinner("Construyendo la red..."):
            network_html = get_character_network(subtitles_path_ner, ner_path)
            st.session_state["network"] = network_html

with col4:
    if "network" in st.session_state:
        st.components.v1.html(st.session_state["network"], height=600, scrolling=True)

# =======================
# 📌 Sección 3: Clasificación de Texto con LLMs
# =======================

st.header("🤖 Clasificación de Texto con LLMs")
col5, col6 = st.columns([1, 2])

with col5:
    text_classification_model = st.text_input("Ruta del Modelo", "models/model.pth")
    text_classification_data_path = st.text_input("Ruta de Datos", "data/classification_data.csv")
    text_to_classify = st.text_area("Texto a Clasificar", "Escribe aquí el texto...")

    if st.button("🔍 Clasificar Texto"):
        with st.spinner("Clasificando..."):
            output = classify_text(text_classification_model, text_classification_data_path, text_to_classify)
            st.session_state["classification"] = output

with col6:
    if "classification" in st.session_state:
        st.text_area("Resultado de la Clasificación", st.session_state["classification"], height=200)

st.success("✅ ¡Análisis Completado! 🎉")
