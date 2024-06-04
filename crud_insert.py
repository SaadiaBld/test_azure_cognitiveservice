from pymongo import MongoClient

# Replace the uri string with your MongoDB deployment's connection string.
uri = "mongodb+srv://saadiabouloudene:vsalFT6Gjz1NUeVr@cluster0.bvccpnu.mongodb.net/"
#mongodb+srv://saadiabouloudene:azertyuiop@cluster0.bvccpnu.mongodb.net/

client = MongoClient(uri)

# database and collection code goes here
db = client['test_speech']
collection = db['nlp']

# insert code goes here
# display the results of your operation

# Close the connection to MongoDB when you're done.
client.close()
