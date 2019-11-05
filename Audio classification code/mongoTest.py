import pymongo
from pymongo import MongoClient
mongoClient = MongoClient('localhost',27017)
db=mongoClient['coughTracker']
user_collection=db.users

data = {"clientId":"111","coughCount":"12"}

user_collection.insert_one(data)

for x in user_collection.find({"clientId":"111"}):
    print(x["clientId"])