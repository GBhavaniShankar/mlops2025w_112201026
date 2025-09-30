from pymongo import MongoClient
import pprint

# MongoDB Atlas Connection
client = MongoClient(
    "mongodb+srv://112201026_db_user:pDnrxcx9EockHy72@cluster0.ogb9zul.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)

db = client["retail_db"]
cust_col = db["customer_centric"]

# Fetch & Print

# Count total documents
count = cust_col.count_documents({})
print(f"Total customer-centric documents in cluster: {count}")

# Fetch one sample document
print("\nSample document:")
doc = cust_col.find_one({})
pprint.pprint(doc)

# Fetch first 2 customers from India (if any exist)
print("\nFirst 2 customers from India:")
for d in cust_col.find({"country": "India"}).limit(2):
    pprint.pprint(d)
