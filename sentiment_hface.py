#transcription de l'audio en texte, analyse des sentiments orchestré par langchain 
### version avec huggingface qui prend input en anglais

from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
from transformers import pipeline
import pymongo
import os
from dotenv import load_dotenv 

load_dotenv() #charger les variables d'environnement à partir du fichier .env 

#recuperer les variables d'environnement
endpoint = os.environ["AZURE_LANGUAGE_ENDPOINT"]
key = os.environ["AZURE_LANGUAGE_KEY"]

#crée un objet azurekeycredential qui contient la clé d'abonnement et un objet textanalyticsclient qui contient l'endpoint et la clé
credential = AzureKeyCredential(key)
text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=credential)

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

# Analyze sentiment par défaut c'est le nouveau modele qui est utilisé
result = text_analytics_client.analyze_sentiment([document['text']], show_opinion_mining=True, show_stats=True, language='fr')
doc = result[0]

#si le sentiment est négatif, apppliquer un modele de LLM mistralai/Codestral pour identifier la raison du sentiment négatif
from transformers import pipeline
from transformers import PreTrainedTokenizerFast
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "distilbert/distilgpt2"
token = os.getenv('TOKEN')
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=token)    
model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=token)

if doc.sentiment == 'negative':
    print("Negative sentiment detected. Identifying reasons...")
    #model = pipeline('text-generation', model='mistralai/Codestral-22B-v0.1', token=os.getenv('TOKEN'))
    text = document['text']
    inputs = tokenizer(text, return_tensors="pt")

    #tokenizer = PreTrainedTokenizerFast.from_pretrained('mistralai/Codestral-22B-v0.1')
    outputs = model.generate(**inputs, max_new_tokens=20)
    prompt = 'The following text has a negative sentiment: ' + text + " The reasons for this sentiment are: "
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    #outputs = model(prompt, max_new_tokens=20)
    print("Raisons identifiées:", tokenizer.decode(outputs[0], skip_special_tokens=True))


if not doc.is_error:
    print(f"Document text: {document['text']}")
    print(f"Overall sentiment: {doc.sentiment}")
    print(f"Confidence scores: positive={doc.confidence_scores.positive:.2f}; neutral={doc.confidence_scores.neutral:.2f}; negative={doc.confidence_scores.negative:.2f}")

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
