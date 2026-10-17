import streamlit as st
from faster_whisper import WhisperModel
import tempfile
import os

st.set_page_config(page_title="BE Video Intercom Transcriptor", page_icon="🎙️", layout="centered")

st.title("🎙️ BE Video Intercom Transcriptor")
st.markdown("### Trascrizione Offline in Tempo Reale")

@st.cache_resource
def load_whisper_model():
    with st.spinner("Caricamento del modello Whisper in corso... Attendere."):
        return WhisperModel("base", device="cpu", compute_type="int8")

model = load_whisper_model()
st.success("Modello pronto e caricato offline!")

st.markdown("---")
st.info("💡 **Come usarlo**: Clicca sul pulsante del microfono qui sotto. Il browser ti chiederà l'accesso al microfono (clicca su *Consenti*). Una volta registrato l'audio, la trascrizione apparirà all'istante.")

audio_value = st.audio_input("Registra dall'intercom / microfono")

if audio_value is not None:
    st.audio(audio_value)
    
    with st.spinner("Trascrizione in corso..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_value.getvalue())
            temp_path = f.name
        
        try:
            segments, info = model.transcribe(temp_path, language="it", beam_size=1)
            transcription_text = "".join([seg.text for seg in segments])
            
            st.subheader("Testo Trascritto:")
            st.text_area("Risultato:", value=transcription_text, height=150)
            
        except Exception as e:
            st.error(f"Errore durante la trascrizione: {e}")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
