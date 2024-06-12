#from msilib import sequence
import azure.cognitiveservices.speech as speechsdk
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
from dotenv import load_dotenv
#from langchain.llms import AzureOpenAI
#from langchain_openai import AzureOpenAI
from langchain.prompts import PromptTemplate
#from langchain.chains import LLMChain, SequentialChain, RunnableSequence
from langchain_core.runnables import RunnableLambda
from concurrent.futures import ThreadPoolExecutor
import pymongo
import openai
import os
import asyncio


load_dotenv()

executor = ThreadPoolExecutor(max_workers=1)

#fonction qui permet de 
async def transcription_audio_texte(audio): 
    if audio is None:
        print("Audio file not yet created")
        return None
    else:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, synchronous_transcription, audio)
        return result
    
def transcription_audio_texte_terminal(_): #fonction qui permet d'engregistrer un vocal en ligne de commandes mais gradio ne peut accéder directement au micro
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

def synchronous_transcription(audio): #fonction pour enregistrer audio avec gradio 
    speech_key, service_region = os.getenv('SPEECH_KEY'), os.getenv('SERVICE_REGION')
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_config.speech_recognition_language="fr-FR"
    audio_config = speechsdk.audio.AudioConfig(filename=audio)
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

def analyse_sentiment_enregistrement(doc):
    #connexion à la base de données
    mongo_uri = os.getenv('MONGO_URI')
    db_name = os.getenv('DB_NAME')
    collection_name = os.getenv('COLLECTION_NAME')
    client = pymongo.MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    #récupération du dernier document inséré
    cursor = collection.find().sort([('_id', pymongo.DESCENDING)]).limit(1)
    document = cursor[0]

    #connexion à l'API Azure Text Analytics pour l'analyse de sentiment
    endpoint = os.environ["AZURE_LANGUAGE_ENDPOINT"]
    key = os.environ["AZURE_LANGUAGE_KEY"]
    credential = AzureKeyCredential(key)
    text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=credential)

    #analyse de sentiment
    result = text_analytics_client.analyze_sentiment([document['text']], show_opinion_mining=True, show_stats=True, language='fr')
    doc = result[0]
    #print('$$$$$$$$$$$$', doc)


    #enregistrement du sentiment dans mongodb
    collection.update_one(
    {'_id': document['_id']}, 
    {'$set':{'sentiment':doc.sentiment, 'confidence_scores':
             {'positive': doc.confidence_scores.positive, 
              'negative': doc.confidence_scores.negative,
              'neutral': doc.confidence_scores.neutral}}})
    print("Sentiment analysis completed and saved to the database")



def if_negative_sentiment(transcription):
    #connexion à la base de données
    mongo_uri = os.getenv('MONGO_URI')
    db_name = os.getenv('DB_NAME')
    collection_name = os.getenv('COLLECTION_NAME')
    client = pymongo.MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    #récupération du dernier document inséré
    cursor = collection.find().sort([('_id', pymongo.DESCENDING)]).limit(1)
    doc = cursor[0]

    if doc and doc['sentiment'] == 'negative':
        openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT") # your endpoint should look like the following
        openai.api_type = 'azure'
        openai.api_version = '2023-05-15'
        deployment_name="QSGC-gpt35"

        prompt=f"Le client a exprimé son insatisfaction dans le texte suivant: '{doc['text']}'. En utilisant le texte, explique en une phrase pourquoi le client est mécontent."
    
        response = openai.ChatCompletion.create(engine=deployment_name, messages=[
                    {"role": "system", "content": "Tu es un assistant utile."},
                    {"role": "user", "content": prompt}], max_tokens=12)
        reason = response['choices'][0]['message']['content'].replace('\n', '').replace(' .', '.').strip()
        print(f"Voici la/les raisons qui expliquent l'insatisfaction du client:", reason)


# Crée une fonction asynchrone
async def main():
    runnable1=RunnableLambda(transcription_audio_texte)
    runnable2=RunnableLambda(enregistrer_texte_mongodb)
    runnable3=RunnableLambda(analyse_sentiment_enregistrement)
    runnable4=RunnableLambda(if_negative_sentiment)
    sequence = runnable1 | runnable2 | runnable3 | runnable4
    await sequence.ainvoke(None) 
    #await llm.analyze_sentiment()

# Exécute la fonction asynchrone
asyncio.run(main())