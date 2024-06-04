#importer librairie speech SDK qui sert à la reconnaissance vocale voir https://github.com/Azure-Samples/cognitive-services-speech-sdk/tree/master/quickstart/python/from-microphone
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import pymongo
import os

load_dotenv()
# créer une variable speech_key qui contient la clé d'abonnement et une variable service_region qui contient la région de service
speech_key, service_region = os.getenv('SPEECH_KEY'), os.getenv('SERVICE_REGION')

# création d'un objet speech_config qui contient la clé d'abonnement et la région de service
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_config.speech_recognition_language="fr-FR"

# créer un objet speech_recognizer qui contient la configuration de reconnaissance vocale
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

print("Je vous écoute...")

# stocker le résultat de la reconnaissance vocale dans une variable result
result = speech_recognizer.recognize_once()

# vérifie si result est une reconnaissance vocale reconnue, non reconnue ou annulée
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

#si reconnaissance vocale valable, insérée dans la base de données mongodb
if recognized_text:
    mongo_uri = os.getenv('MONGO_URI')
    db_name = os.getenv('DB_NAME')
    collection_name = os.getenv('COLLECTION_NAME')

    #se connecte à mongodb
    client = pymongo.MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    #Document à insérer
    document = {"text": recognized_text}    
    result = collection.insert_one(document)
    print("Document inséré avec l'ID {}".format(result.inserted_id))
