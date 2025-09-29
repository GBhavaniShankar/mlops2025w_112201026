import pandas as pd
from pymongo import MongoClient

# Load Data
df = pd.read_csv("data/online_retail.csv")
df = df.dropna(subset=["CustomerID"])

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["retail_db"]

# Collections
transactions = db["transaction_centric"]
customers = db["customer_centric"]

# Transaction-centric Data
grouped_txn = df.groupby(["InvoiceNo", "CustomerID", "InvoiceDate", "Country"])

txn_docs = []
for (invoice, customer, date, country), group in grouped_txn:
    items = group[["StockCode", "Description", "Quantity", "UnitPrice"]].to_dict("records")
    doc = {
        "invoice_no": str(invoice),
        "customer_id": str(int(customer)),
        "invoice_date": pd.to_datetime(date).isoformat(),
        "country": country,
        "items": items
    }
    txn_docs.append(doc)

if txn_docs:
    transactions.insert_many(txn_docs)
    print(f"Inserted {len(txn_docs)} transaction documents")

# Customer-centric Data
grouped_cust = df.groupby(["CustomerID", "Country"])

cust_docs = []
for (customer, country), group in grouped_cust:
    invoices = []
    for inv_no, inv_group in group.groupby(["InvoiceNo", "InvoiceDate"]):
        invoice_no, inv_date = inv_no
        items = inv_group[["StockCode", "Description", "Quantity", "UnitPrice"]].to_dict("records")
        invoices.append({
            "invoice_no": str(invoice_no),
            "invoice_date": pd.to_datetime(inv_date).isoformat(),
            "items": items
        })

    doc = {
        "customer_id": str(int(customer)),
        "country": country,
        "invoices": invoices
    }
    cust_docs.append(doc)

if cust_docs:
    customers.insert_many(cust_docs)
    print(f"Inserted {len(cust_docs)} customer documents")
