import gradio as gr
import azure.cognitiveservices.speech as speechsdk
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import pymongo
import openai
import os
import asyncio

load_dotenv()

executor = ThreadPoolExecutor(max_workers=1)

def transcription_audio_texte(audio):
    if audio is None:
        return "Aucun fichier audio fourni"
    speech_key, service_region = os.getenv('SPEECH_KEY'), os.getenv('SERVICE_REGION')
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_config.speech_recognition_language = "fr-FR"

    if isinstance(audio, str):
        audio_path = audio
    else:
        audio_path = audio.name

    if not os.path.isfile(audio_path):
        return "Fichier audio non valide"
    
    audio_config = speechsdk.audio.AudioConfig(filename=audio)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    result = speech_recognizer.recognize_once()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        return "Echec de la reconnaissance vocale"
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        return f"Reconnaissance vocale annulée: {cancellation_details.reason}"

def enregistrer_texte_mongodb(text_transcrit):
    if text_transcrit:
        mongo_uri = os.getenv('MONGO_URI')
        db_name = os.getenv('DB_NAME')
        collection_name = os.getenv('COLLECTION_NAME')
        client = pymongo.MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]
        document = {"text": text_transcrit}
        collection.insert_one(document)

def analyse_sentiment(text):
    endpoint = os.getenv("AZURE_LANGUAGE_ENDPOINT")
    key = os.getenv("AZURE_LANGUAGE_KEY")
    credential = AzureKeyCredential(key)
    text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=credential)

    response = text_analytics_client.analyze_sentiment([text], show_opinion_mining=True, language='fr')
    doc = response[0]
    return doc.sentiment, doc.confidence_scores

def if_negative_sentiment(text):
    openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
    openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai.api_type = 'azure'
    openai.api_version = '2023-05-15'
    deployment_name = "QSGC-gpt35"

    prompt = f"Le client a exprimé son insatisfaction dans le texte suivant: '{text}'. En utilisant le texte, explique en une phrase pourquoi le client est mécontent."

    response = openai.ChatCompletion.create(
        engine=deployment_name,
        messages=[
            {"role": "system", "content": "Tu es un assistant utile."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50
    )
    reason = response['choices'][0]['message']['content'].strip()
    return reason

async def process_audio(audio):
    transcript = transcription_audio_texte(audio)
    enregistrer_texte_mongodb(transcript)
    sentiment, scores = analyse_sentiment(transcript)

    if sentiment == 'negative':
        reason = if_negative_sentiment(transcript)
        return transcript, sentiment, scores, reason
    return transcript, sentiment, scores, None

def gradio_interface(audio):
    # loop = asyncio.get_event_loop()
    # return loop.run_until_complete(process_audio(audio))
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
