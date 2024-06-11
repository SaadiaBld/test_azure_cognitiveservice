#transcription de l'audio en texte, analyse des sentiments orchestré par langchain 
#créer une chaine qui génére la transcription de l'audio en texte et analyse des sentiments 
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
from dotenv import load_dotenv 
import openai
import pymongo
import os
#from langchain import LangChain 

load_dotenv() #charger les variables d'environnement à partir du fichier .env 

#recuperer les variables d'environnement
endpoint = os.environ["AZURE_LANGUAGE_ENDPOINT"]
key = os.environ["AZURE_LANGUAGE_KEY"]

#crée un objet azurekeycredential qui contient la clé d'abonnement et un objet textanalyticsclient qui contient l'endpoint et la clé
credential = AzureKeyCredential(key)
text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=credential)

#connection à mongodb 
mongo_uri = os.getenv('MONGO_URI')
db_name = os.getenv('DB_NAME')
collection_name = os.getenv('COLLECTION_NAME')

client = pymongo.MongoClient(mongo_uri)
db = client[db_name]
collection = db[collection_name]

# obtenir le document le plus récent de la collection
cursor = collection.find().sort([('_id', pymongo.DESCENDING)]).limit(1)
document = cursor[0]

# Analyse sentiment, par défaut c'est le nouveau modele qui est utilisé
result = text_analytics_client.analyze_sentiment([document['text']], show_opinion_mining=True, show_stats=True, language='fr')
doc = result[0]

#enregistrer le sentiment de la conversation dans la base de données 
collection.update_one(
    {'_id': document['_id']}, 
    {'$set':{'sentiment':doc.sentiment, 'confidence_scores':
             {'positive': doc.confidence_scores.positive, 
              'negative': doc.confidence_scores.negative,
              'neutral': doc.confidence_scores.neutral}}})
print("Sentiment analysis completed and saved to the database")


#si le sentiment est négatif, apppliquer modele de azure openAI pour identifier la raison du sentiment négatif 
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT") # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type = 'azure'
openai.api_version = '2023-05-15' 

deployment_name="QSGC-gpt35"
if doc.sentiment == 'negative':
    #print("Negative sentiment detected. Identifying reasons...")
    #prompt = 'Explique à partir de la réponse du client pourquoi il est insatisfait: '
    #response = openai.ChatCompletion.create(engine=deployment_name, prompt=prompt, max_tokens=12)
    #text = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
    #print(f"Voici la/les raisons qui expliquent l'insatisfaction du client:", text)
    prompt = f"Le client a exprimé son insatisfaction dans le texte suivant: '{document['text']}'. En utilisant le texte, explique de façon concise les raisons de cette insatisfaction."
    response = openai.Completion.create(
        engine=deployment_name,
        prompt=prompt,
        max_tokens=100)
    reason_text=response['choices'][0]['text'].strip()
    print("$$$$$$VOICI LE/LES RAISON(S) DE L'INSATISFACTION DU CLIENT", reason_text)

    #enregistrer les raisons identifiées dans la base de données
    collection.update_one(
        {'_id': document['_id']}, 
        {'$set':{'reasons': reason_text}})
    print("Reasons for negative sentiment saved to the database")

if not doc.is_error:
    print(f"Document text: {document['text']}")
    print(f"Overall sentiment: {doc.sentiment}")
    print(f"Confidence scores: positive={doc.confidence_scores.positive:.2f}; neutral={doc.confidence_scores.neutral:.2f}; negative={doc.confidence_scores.negative:.2f}")

else:
    print(f"Error: {doc.error}")



#print last document in the collection 
cursor = collection.find().sort([('_id', pymongo.DESCENDING)]).limit(1)
last_document = cursor[0]
print(f"Last document in the collection: {last_document}")

####integration de langchain pour creer la chaine de traitement de la transcription de l'audio en texte et analyse des sentiments

