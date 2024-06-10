import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import pymongo
import os

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

def transcription_audio_texte():
    # ... (same as before)

def enregistrer_texte_mongodb(transcribed_text):
    if transcribed_text:
        mongo_uri = os.getenv('MONGO_URI')
        db_name = os.getenv('DB_NAME')
        collection_name = os.getenv('COLLECTION_NAME')
        client = pymongo.MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]
        document = {"text": transcribed_text}    
        result = collection.insert_one(document)
        print("Document inséré avec l'ID {}".format(result.inserted_id))

chain = SequentialChain([
    LLMChain(llm, transcription_audio_texte),
    LLMChain(llm, enregistrer_texte_mongodb)
])
chain("")