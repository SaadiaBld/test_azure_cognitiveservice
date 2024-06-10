import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from langchain.llms import AzureOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import pymongo
import os

llm = AzureOpenAI()

def transcription_audio_texte():
    speech_key, service_region = os.getenv('SPEECH_KEY'), os.getenv('SERVICE_REGION')
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_config.speech_recognition_language="fr-FR"
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
    print("Je vous écoute...")

    result = speech_recognizer.recognize_once()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        recognized_text = result.text
        print("Reconnaissance vocale validée: {}".format(result.text))
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("Reconnaissance vocale a échoué: {}".format(result.no_match_details))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Reconnaissance vocale annulée: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
        recognized_text = None
    return recognized_text 

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
        print("Document inséré avec l'ID {}".format(result.inserted_id))


chain=LLMChain([transcription_audio_texte, enregistrer_texte_mongodb])
chain = SequentialChain([LLMChain(llm, transcription_audio_texte), LLMChain(llm, enregistrer_texte_mongodb)])
chain("")