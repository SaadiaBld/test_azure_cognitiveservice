import gradio as gr
import soundfile as sf
import os
from datetime import datetime
from fichiers_brouillons.chaine import synchronous_transcription, analyse_sentiment_enregistrement, if_negative_sentiment

# Fonction pour enregistrer l'audio
def enregistrer_audio(audio):
    audio = gr.Interface.load()

    if audio is None:
        return "Aucun fichier audio enregistré."

    # le chemin pour enregistrer l'audio
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    audio_path = f"audio_{timestamp}.wav"
    
    # Sauvegarder l'audio
    sf.write(audio_path, audio, 44100)

def process_audio(audio): 
    # Analyser l'audio avec une fonction NLP
    if audio is not None:
        transcription = synchronous_transcription(audio)
        sentiment = analyse_sentiment_enregistrement(transcription)
        negative_sentiment = if_negative_sentiment(transcription)

    return transcription, sentiment, negative_sentiment


# Définition de l'interface Gradio
with gr.Blocks() as interface:
    with gr.Row():
        audio_input = gr.Microphone(label="Enregistrer votre audio") as audio
        save_button = gr.Button("Enregistrer et Analyser")
    output = gr.Textbox(label="Résultat de la transcription") as process_audio()

    save_button.click(enregistrer_audio, inputs=audio_input, outputs=output)

# Lancement de l'interface
interface.launch()
