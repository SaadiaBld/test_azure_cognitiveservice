from process_audio_functions import transcription_audio_texte, analyse_sentiment, if_negative_sentiment, enregistrer_texte_mongodb
import gradio as gr
import asyncio


async def process_audio(audio):
    transcript = transcription_audio_texte(audio)
    enregistrer_texte_mongodb(transcript)
    sentiment, scores = analyse_sentiment(transcript)

    if sentiment == 'negative':
        reason = if_negative_sentiment(transcript)
        return transcript, sentiment, scores, reason
    return transcript, sentiment, scores, None

def gradio_interface(audio):
    return asyncio.run(process_audio(audio))

interface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Audio(type="filepath"),
    outputs=[
        gr.Textbox(label="Transcription"),
        gr.Textbox(label="Sentiment"),
        gr.Textbox(label="Confidence Scores"),
        gr.Textbox(label="Raison de l'insatisfaction")
    ],
    title="Analyse de Sentiment d'Audio",
    description="Enregistrez un audio ou glissez-déposez un fichier audio pour le transcrire et analyser le sentiment. Si le sentiment est négatif, la raison sera affichée."
)

interface.launch()
