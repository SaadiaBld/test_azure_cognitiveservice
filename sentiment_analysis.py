#transcription de l'audio en texte, analyse des sentiments orchestré par langchain 
#créer une chaine qui génére la transcription de l'audio en texte et analyse des sentiments 
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
import pymongo
#from langchain import LangChain
import os

#recuperer les variables d'environnement
endpoint = os.environ["AZURE_LANGUAGE_ENDPOINT"]
key = os.environ["AZURE_LANGUAGE_KEY"]

#crée un objet azurekeycredential qui contient la clé d'abonnement et un objet textanalyticsclient qui contient l'endpoint et la clé
credential = AzureKeyCredential(os.environ["AZURE_LANGUAGE_KEY"])
text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))

#récupérer le document le plus récent de la collection
#connect to mongodb 
mongo_uri = os.getenv('MONGO_URI')
db_name = os.getenv('DB_NAME')
collection_name = os.getenv('COLLECTION_NAME')

client = pymongo.MongoClient(mongo_uri)
db = client[db_name]
collection = db[collection_name]

# Fetch the most recent document from the collection
cursor = collection.find().sort([('_id', pymongo.DESCENDING)]).limit(1)
document = cursor[0]

# Analyze sentiment
result = text_analytics_client.analyze_sentiment([document['text']], show_opinion_mining=True, show_stats=True)
doc = result[0]

if not doc.is_error:
    print(f"Document text: {document['text']}")
    print(f"Overall sentiment: {doc.sentiment}")
    print(f"Confidence scores: positive={doc.confidence_scores.positive:.3f}; neutral={doc.confidence_scores.neutral:.3f}; negative={doc.confidence_scores.negative:.3f}")

else:
    print(f"Error: {doc.error}")

#add new field to the document in the collection 
collection.update_one(
    {'_id': document['_id']}, 
    {'$set':{'sentiment':doc.sentiment, 'confidence_scores':
             {'positive': doc.confidence_scores.positive, 
              'negative': doc.confidence_scores.negative,
              'neutral': doc.confidence_scores.neutral}}})
print("Sentiment analysis completed and saved to the database")

#print last document in the collection 
cursor = collection.find().sort([('_id', pymongo.DESCENDING)]).limit(1)
last_document = cursor[0]
print(f"Last document in the collection: {last_document}")