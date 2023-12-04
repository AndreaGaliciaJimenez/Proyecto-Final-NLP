import streamlit as st
import openai
from textblob import TextBlob
import librosa.display
import matplotlib.pyplot as plt

# Clave API de OpenAI
mi_clave_api = "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"  
st.title("Resumidor de Audio")

clave = mi_clave_api
cliente_openai = openai.OpenAI(api_key=clave)

# Ruta del archivo de audio
archivo_audio = '/Users/andreagalicia/Desktop/Whisper-ChatGPT-Audio/MA1.m4a'  

# Función para transcribir audio utilizando OpenAI
def transcribir_audio(archivo_audio):
    with open(archivo_audio, "rb") as audio_file:
        transcripcion = cliente_openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcripcion.text if transcripcion else "No se pudo transcribir el audio."

# Función para análisis de sentimientos utilizando TextBlob
def analizar_sentimientos(texto):
    blob = TextBlob(texto)
    sentimiento = blob.sentiment.polarity
    if sentimiento > 0.1:
        return "Positivo"
    elif sentimiento < -0.1:
        return "Negativo"
    else:
        return "Neutral"

# Reproducción de audio
audio_bytes = open(archivo_audio, 'rb').read()
st.audio(audio_bytes, format='audio/mpeg')

# Transcripción de audio
st.subheader("Transcripción:")
texto_transcrito = transcribir_audio(archivo_audio)
st.write(texto_transcrito)

# Resumen del audio (utilizando OpenAI GPT-3)
messages = [
    {"role": "system", "content": "Resumen del audio, por favor."},
    {"role": "user", "content": texto_transcrito}
]
response = cliente_openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)
resumen_audio = response.choices[0].message.content
st.subheader("Resumen:")
st.write(resumen_audio)


# Análisis de sentimientos en la transcripción
st.subheader("Análisis de Sentimientos:")
sentimiento = analizar_sentimientos(texto_transcrito)
st.write(f"Sentimiento: {sentimiento}")


