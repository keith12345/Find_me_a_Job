from pymongo import MongoClient


client = MongoClient()

db = client['jobs_data']

db.create_collection('Chicago_Jobs')
db.create_collection('Los_Angeles_Jobs')
db.create_collection('New_York_Jobs')
db.create_collection('SF_Bay_Area_Jobs')
db.create_collection('Seattle_Jobs')