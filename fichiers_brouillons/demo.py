#construire une demo avec gradio 
import gradio as gr
import asyncio
import shutil
import tempfile
from fichiers_brouillons.chaine import transcription_audio_texte, analyse_sentiment_enregistrement, if_negative_sentiment



# async def process_audio(audio):
#     # Transcription de l'audio
#     transcription = await transcription_audio_texte(audio)
#     # Analyse de sentiment
#     sentiment = analyse_sentiment_enregistrement(transcription)
#     # Vérification si le sentiment est négatif
#     negative_sentiment = if_negative_sentiment(transcription)
#     return transcription, sentiment, negative_sentiment

# # Création de l'interface Gradio
# interface = gr.Interface(
#     fn=process_audio,
#     inputs=gr.Audio(type="filepath", label="Upload audio file"), 
#     outputs=[
#         gr.Textbox(label="Transcription"), 
#         gr.Textbox(label="Sentiment"),
#         gr.Textbox(label="Negative Sentiment")
#     ]
# )

# interface.launch()

def process_audio(audio):
    # Transcrire l'audio en texte
    text_transcrit = transcription_audio_texte(audio)
    
    # Analyser le sentiment de l'enregistrement
    sentiment = analyse_sentiment_enregistrement(text_transcrit)
    
    # Si le sentiment est négatif, obtenir la raison
    if sentiment == 'negative':
        reason = if_negative_sentiment(text_transcrit)
    else:
        reason = "Sentiment is not negative"
    
    return sentiment, reason

# Définition de l'interface Gradio
with gr.Blocks() as interface:
    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Enregistrer votre audio")
        process_button = gr.Button("Enregistrer et Analyser")
    sentiment_output = gr.Textbox(label="Sentiment")
    reason_output = gr.Textbox(label="Raison du sentiment négatif")

    process_button.click(process_audio, inputs=audio_input, outputs=[sentiment_output, reason_output])

# Lancement de l'interface
interface.launch()


