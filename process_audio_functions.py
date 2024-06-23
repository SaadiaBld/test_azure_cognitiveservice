import gradio as gr
import azure.cognitiveservices.speech as speechsdk
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from concurrent.futures import ThreadPoolExecutor
import pymongo
import openai
import os
import asyncio

load_dotenv()

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
    """fonction asynchrone qui execute les fonctions de traitement de l'audio 
    pour éviter le blocage de l'interface utilisateur pendant le traitement de l'audio. 
    Chaque étape est éxécutée de façon séquentielle. Cette fonction est appelée par la fonction gradio_interface.
    Gradio s'attend à ce que la fonction de traitement de l'audio soit synchrone pour gérer input et outputs.
    Chaque fonction est transformée en tache asynchrone avec runnablelabda """

    runnable1 = RunnableLambda(transcription_audio_texte)
    runnable2 = RunnableLambda(enregistrer_texte_mongodb)
    runnable3 = RunnableLambda(analyse_sentiment)
    runnable4 = RunnableLambda(if_negative_sentiment)
    
    # exécution de chaque fonction de traitement de l'audio de façon séquentielle 
    transcription = await runnable1.ainvoke(audio) # transcription_audio_texte
    await runnable2.ainvoke(transcription) # enregistrer_texte_mongodb
    sentiment, scores = await runnable3.ainvoke(transcription) # analyse_sentiment
    
    if sentiment == 'negative': #si sentiment négatif, exécute la fonction if_negative_sentiment
        reason = await runnable4.ainvoke(transcription)
        return transcription, sentiment, scores, reason
    return transcription, sentiment, scores, None #si pas négatif, retourne les résultats et None pour la raison
