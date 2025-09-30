import pandas as pd
from pymongo import MongoClient

# MongoDB Atlas Connection
client = MongoClient(
    "mongodb+srv://112201026_db_user:pDnrxcx9EockHy72@cluster0.ogb9zul.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)

db = client["retail_db"]
cust_col = db["customer_centric"]

# Drop existing collection (start fresh)
cust_col.drop()

# Load CSV (first 1000 rows)
df = pd.read_csv("data/online_retail.csv")
df = df.dropna(subset=["CustomerID"])
df = df.iloc[:1000]

# Build Customer-Centric Docs
cust_docs = []
grouped = df.groupby(["CustomerID", "Country"])

for (customer, country), group in grouped:
    invoices = []
    total_spent = 0
    for (inv_no, inv_date), g in group.groupby(["InvoiceNo", "InvoiceDate"]):
        items = g[["StockCode", "Description", "Quantity", "UnitPrice"]].to_dict("records")
        total = (g["Quantity"] * g["UnitPrice"]).sum()
        total_spent += total
        invoices.append({
            "invoice_no": str(inv_no),
            "invoice_date": pd.to_datetime(inv_date).isoformat() if not pd.isna(inv_date) else None,
            "total_amount": round(total, 2),
            "items": items
        })
    cust_docs.append({
        "customer_id": str(int(customer)),
        "country": country,
        "total_spent": round(total_spent, 2),
        "invoices": invoices
    })

# Insert into Atlas
if cust_docs:
    cust_col.insert_many(cust_docs)
    print(f"Inserted {len(cust_docs)} customer-centric documents into MongoDB Atlas.")
else:
    print("No documents to insert.")
