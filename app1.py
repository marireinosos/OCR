"""
📸 Lector Mágico de Mari — Reconocimiento óptico de caracteres (OCR)
Toma una foto o sube una imagen y la app lee el texto que tiene.
"""

import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image


# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Lector Mágico de Mari",
    page_icon="📸",
    layout="wide",
)


def html(codigo):
    """Muestra HTML/CSS. Usa st.html (Streamlit nuevo) y si no existe, st.markdown."""
    if hasattr(st, "html"):
        st.html(codigo)
    else:
        st.markdown(codigo, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ESTILOS ROSADOS 💗
# ─────────────────────────────────────────────
html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Pacifico&display=swap');

    .stApp, .stApp p, .stApp label, .stApp input, .stApp textarea,
    .stApp button, .stApp h1, .stApp h2, .stApp h3, .stApp li {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #fff0f6 0%, #ffe4ef 100%);
    }

    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 2px dashed #f9a8d4;
    }
    [data-testid="stSidebar"] h2 {
        font-family: 'Pacifico', cursive !important;
        color: #db2777 !important;
        font-weight: 400 !important;
    }
    [data-testid="stSidebar"] h3 {
        color: #be185d !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }

    h1, h2, h3 { color: #9d174d !important; }
    p, li, label { color: #6b2143; }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #fbcfe8 !important;
        border-radius: 24px !important;
        background: #ffffff;
        box-shadow: 0 6px 20px rgba(236,72,153,0.08);
    }

    .stButton > button, [data-testid="stDownloadButton"] button {
        background: linear-gradient(135deg, #ec4899, #f472b6) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(236,72,153,0.35) !important;
        transition: transform 0.2s ease !important;
    }
    .stButton > button:hover, [data-testid="stDownloadButton"] button:hover {
        transform: translateY(-2px) scale(1.02);
    }

    textarea {
        background: #fffafc !important;
        border: 2px solid #fbcfe8 !important;
        border-radius: 16px !important;
        color: #500724 !important;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 2px solid #fbcfe8;
        border-radius: 20px;
        padding: 14px 18px;
    }
    [data-testid="stMetricValue"] { color: #db2777 !important; }

    .header-card {
        background: linear-gradient(135deg, #f472b6 0%, #ec4899 50%, #c084fc 100%);
        border-radius: 28px;
        padding: 36px 40px;
        margin-bottom: 10px;
        box-shadow: 0 12px 30px rgba(236,72,153,0.30);
        text-align: center;
        position: relative;
        overflow: hidden;
        font-family: 'Poppins', sans-serif;
    }
    .header-card h1 {
        font-family: 'Pacifico', cursive;
        color: #ffffff;
        font-weight: 400;
        font-size: 2.6rem;
        margin: 0;
    }
    .header-card p {
        color: #fff0f6;
        font-size: 1.05rem;
        margin: 8px 0 0 0;
    }
    .flota {
        position: absolute;
        font-size: 1.5rem;
        opacity: 0.6;
        animation: flotar 4s ease-in-out infinite;
    }
    @keyframes flotar {
        0%, 100% { transform: translateY(0) rotate(0deg); }
        50%      { transform: translateY(-10px) rotate(8deg); }
    }

    .pasos {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        font-family: 'Poppins', sans-serif;
    }
    .paso {
        flex: 1;
        min-width: 150px;
        background: #ffffff;
        border: 2px solid #fbcfe8;
        border-radius: 20px;
        padding: 16px;
        text-align: center;
        color: #9d174d;
        font-size: 0.9rem;
        transition: transform 0.2s;
    }
    .paso:hover { transform: translateY(-4px); }
    .paso b { display: block; font-size: 1.6rem; margin-bottom: 4px; }

    .texto-leido {
        background: #fdf2f8;
        border: 2px dashed #f472b6;
        border-radius: 20px;
        padding: 20px 24px;
        color: #500724;
        font-family: 'Poppins', sans-serif;
        font-size: 1rem;
        line-height: 1.8;
        white-space: pre-wrap;
    }

    .footer {
        text-align: center;
        color: #db2777;
        font-size: 0.9rem;
        padding: 24px 0 6px 0;
        font-family: 'Poppins', sans-serif;
    }
</style>
""")


# ─────────────────────────────────────────────
# FILTROS (cómo se limpia la foto antes de leerla)
# ─────────────────────────────────────────────
FILTROS = {
    "✨ Automático (recomendado)": "auto",
    "🌗 Invertir colores": "invertir",
    "🩶 Blanco y negro": "gris",
    "🖤 Alto contraste": "contraste",
    "📷 Sin filtro": "ninguno",
}

IDIOMAS = {
    "Español 🇨🇴": "spa",
    "Inglés 🇺🇸": "eng",
    "Español + Inglés": "spa+eng",
}


def aplicar_filtro(img_bgr, filtro):
    """Recibe la imagen de OpenCV (BGR) y devuelve la imagen lista para leer."""
    if filtro == "ninguno":
        return img_bgr

    if filtro == "invertir":
        return cv2.bitwise_not(img_bgr)

    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if filtro == "gris":
        return gris

    if filtro == "contraste":
        _, bn = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return bn

    # "auto": agranda, quita ruido y deja el texto negro sobre blanco
    gris = cv2.resize(gris, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    gris = cv2.GaussianBlur(gris, (3, 3), 0)
    bn = cv2.adaptiveThreshold(
        gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    # si la mayoría de la imagen es oscura, invertimos para que el fondo sea blanco
    if np.mean(bn) < 127:
        bn = cv2.bitwise_not(bn)
    return bn


def leer_texto(img, idioma):
    """Lee el texto con Tesseract. Si el idioma no está instalado, usa inglés."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        return pytesseract.image_to_string(img, lang=idioma), idioma
    except pytesseract.TesseractError:
        return pytesseract.image_to_string(img, lang="eng"), "eng"


def a_rgb(img):
    """Convierte para mostrar en Streamlit."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────
# PANEL LATERAL
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Lector Mágico")
    st.caption("Fotos que se vuelven texto ✨")
    st.divider()

    st.markdown("### 📥 ¿De dónde sale la imagen?")
    fuente = st.radio("fuente", ["📸 Tomar foto", "🖼️ Subir imagen"],
                      label_visibility="collapsed")

    st.divider()
    st.markdown("### 🎨 Filtro")
    filtro_sel = st.radio("filtro", list(FILTROS.keys()), label_visibility="collapsed")

    st.divider()
    st.markdown("### 🌎 Idioma del texto")
    idioma_sel = st.selectbox("idioma", list(IDIOMAS.keys()), label_visibility="collapsed")

    st.divider()
    st.caption("💡 Tip: buena luz, texto derecho y sin sombras = mejor lectura.")


# ─────────────────────────────────────────────
# ENCABEZADO
# ─────────────────────────────────────────────
html("""
<div class="header-card">
    <span class="flota" style="top:18px; left:30px;">📸</span>
    <span class="flota" style="top:55px; right:50px; animation-delay:1s;">✨</span>
    <span class="flota" style="bottom:16px; left:22%; animation-delay:2s;">💗</span>
    <span class="flota" style="bottom:20px; right:24%; animation-delay:1.5s;">📝</span>
    <h1>Lector Mágico de Mari</h1>
    <p>Toma una foto de cualquier texto y lo convierto en palabras que puedes copiar 💌</p>
</div>
""")

html("""
<div class="pasos">
    <div class="paso"><b>📸</b>1. Toma o sube una foto</div>
    <div class="paso"><b>🎨</b>2. Elige un filtro</div>
    <div class="paso"><b>🔮</b>3. La app lee el texto</div>
    <div class="paso"><b>💾</b>4. Cópialo o descárgalo</div>
</div>
""")

st.write("")


# ─────────────────────────────────────────────
# ENTRADA DE IMAGEN
# ─────────────────────────────────────────────
if fuente == "📸 Tomar foto":
    archivo = st.camera_input("Toma una foto 📸")
else:
    archivo = st.file_uploader("Sube una imagen 🖼️", type=["png", "jpg", "jpeg", "webp"])

if archivo is None:
    st.info("💗 Toma una foto o sube una imagen para empezar.")
    html('<div class="footer">Hecho con 💗 por Mari</div>')
    st.stop()


# ─────────────────────────────────────────────
# PROCESAMIENTO
# ─────────────────────────────────────────────
bytes_data = archivo.getvalue()
img_original = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

if img_original is None:
    st.error("No pude abrir esa imagen 😢 Prueba con otra (PNG o JPG).")
    st.stop()

img_filtrada = aplicar_filtro(img_original, FILTROS[filtro_sel])

with st.spinner("Leyendo tu imagen con magia 🔮..."):
    texto, idioma_usado = leer_texto(img_filtrada, IDIOMAS[idioma_sel])

texto = texto.strip()

if idioma_usado != IDIOMAS[idioma_sel]:
    st.warning("El idioma elegido no está instalado en el servidor, así que leí en inglés. "
               "Revisa el archivo packages.txt 💡")


# ── Antes y después ──
col1, col2 = st.columns(2, gap="large")
with col1:
    with st.container(border=True):
        st.markdown("### 🖼️ Foto original")
        st.image(a_rgb(img_original), use_container_width=True)
with col2:
    with st.container(border=True):
        st.markdown(f"### ✨ Con filtro")
        st.caption(filtro_sel)
        st.image(a_rgb(img_filtrada), use_container_width=True)

st.write("")

# ── Resultado ──
if not texto:
    st.warning("No encontré texto 🥲 Prueba otro filtro, acércate más o mejora la luz.")
else:
    palabras = texto.split()
    m1, m2, m3 = st.columns(3)
    m1.metric("📝 Palabras", len(palabras))
    m2.metric("🔤 Caracteres", len(texto))
    m3.metric("📄 Líneas", len([l for l in texto.splitlines() if l.strip()]))

    st.write("")
    with st.container(border=True):
        st.markdown("### 💌 Texto encontrado")
        texto_seguro = texto.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html(f'<div class="texto-leido">{texto_seguro}</div>')

        st.write("")
        st.caption("Para copiarlo, pasa el mouse por el recuadro de abajo y dale al ícono de copiar 📋")
        st.code(texto, language=None)

        st.download_button(
            "💾 Descargar texto (.txt)",
            data=texto.encode("utf-8"),
            file_name="texto_leido_mari.txt",
            mime="text/plain",
            use_container_width=True,
        )

html('<div class="footer">Hecho con 💗 por Mari</div>')
    


    


