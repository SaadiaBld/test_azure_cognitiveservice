import os
import gradio as gr
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import pymongo
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

# Charger les variables d'environnement à partir du fichier .env
load_dotenv()

def enregistrer_texte_mongodb(text_transcrit):
    if text_transcrit:
        mongo_uri = os.getenv('MONGO_URI')
        db_name = os.getenv('DB_NAME')
        collection_name = os.getenv('COLLECTION_NAME')
        client = pymongo.MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]
        document = {"text": text_transcrit}    
        result = collection.insert_one(document)
        return result.inserted_id

def analyse_sentiment_enregistrement(document_id):
    #connexion à la base de données
    mongo_uri = os.getenv('MONGO_URI')
    db_name = os.getenv('DB_NAME')
    collection_name = os.getenv('COLLECTION_NAME')
    client = pymongo.MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    #récupération du document par ID
    document = collection.find_one({'_id': document_id})

    #connexion à l'API Azure Text Analytics pour l'analyse de sentiment
    endpoint = os.getenv("AZURE_LANGUAGE_ENDPOINT")
    key = os.getenv("AZURE_LANGUAGE_KEY")
    credential = AzureKeyCredential(key)
    text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=credential)

    #analyse de sentiment
    result = text_analytics_client.analyze_sentiment([document['text']], show_opinion_mining=True, show_stats=True, language='fr')
    doc = result[0]

    #enregistrement du sentiment dans mongodb
    collection.update_one(
        {'_id': document['_id']}, 
        {'$set': {
            'sentiment': doc.sentiment, 
            'confidence_scores': {
                'positive': doc.confidence_scores.positive, 
                'negative': doc.confidence_scores.negative,
                'neutral': doc.confidence_scores.neutral
            }
        }}
    )
    return {
        "sentiment": doc.sentiment,
        "confidence_scores": {
            "positive": doc.confidence_scores.positive,
            "negative": doc.confidence_scores.negative,
            "neutral": doc.confidence_scores.neutral
        }
    }

def synchronous_transcription(audio):
    speech_key, service_region = os.getenv('SPEECH_KEY'), os.getenv('SERVICE_REGION')
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_config.speech_recognition_language = "fr-FR"

    if audio is None:
        return "Aucun fichier audio fourni."

    if isinstance(audio, str):
        audio_path = audio
    else:
        audio_path = audio.name

    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    result = speech_recognizer.recognize_once()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        recognized_text = result.text
        # Enregistrer le texte transcrit dans MongoDB
        document_id = enregistrer_texte_mongodb(recognized_text)
        # Analyser le sentiment
        sentiment_result = analyse_sentiment_enregistrement(document_id)
        return {
            "Transcription": recognized_text,
            "Sentiment": sentiment_result
        }
    elif result.reason == speechsdk.ResultReason.NoMatch:
        return "Echec de la reconnaissance vocale: {}".format(result.no_match_details)
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        error_message = "Reconnaissance vocale annulée: {}".format(cancellation_details.reason)
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            error_message += "\nError details: {}".format(cancellation_details.error_details)
        return error_message

# Définition de l'interface Gradio
with gr.Blocks() as interface:
    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Enregistrer votre audio")
        save_button = gr.Button("Enregistrer et Analyser")
    transcription_output = gr.Textbox(label="Transcription")
    sentiment_output = gr.JSON(label="Résultat de l'analyse de sentiment")

    save_button.click(synchronous_transcription, inputs=audio_input, outputs=[transcription_output, sentiment_output])

# Lancement de l'interface
interface.launch(share=True)
