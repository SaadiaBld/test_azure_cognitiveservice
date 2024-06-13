import os
import gradio as gr
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

# Charger les variables d'environnement à partir du fichier .env
load_dotenv()

def synchronous_transcription(audio):
    speech_key, service_region = os.getenv('SPEECH_KEY'), os.getenv('SERVICE_REGION')
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_config.speech_recognition_language = "fr-FR"

    if audio is None:
        return "Aucun fichier audio fourni."

    #audio_path = audio.name
    if isinstance(audio, str):
        audio_path = audio
    else:
        audio_path = audio.name
    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    result = speech_recognizer.recognize_once()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        recognized_text = result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        recognized_text = "Echec de la reconnaissance vocale: {}".format(result.no_match_details)
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        recognized_text = "Reconnaissance vocale annulée: {}".format(cancellation_details.reason)
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            recognized_text += "\nError details: {}".format(cancellation_details.error_details)
    return recognized_text

# Définition de l'interface Gradio
with gr.Blocks() as interface:
    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Enregistrer votre audio")
        save_button = gr.Button("Enregistrer et Analyser")
    output = gr.Textbox(label="Transcription")

    save_button.click(synchronous_transcription, inputs=audio_input, outputs=output)

# Lancement de l'interface
interface.launch()
