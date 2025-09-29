import pandas as pd
import time
from pymongo import MongoClient
import matplotlib.pyplot as plt
import numpy as np
import copy

# ============================================================
# Load sample data
# ============================================================
df = pd.read_csv("data/online_retail.csv")
df = df.dropna(subset=["CustomerID"])
df = df.iloc[1000:2000]   # just taking 1000 rows to keep it light

# ============================================================
# Connect to Mongo
# ============================================================
client = MongoClient("mongodb://localhost:27017/")
db = client["retail_db"]

txn_col = db["transaction_centric"]
cust_col = db["customer_centric"]

# clear old collections (if any)
txn_col.drop()
cust_col.drop()

# a couple of useful indexes
txn_col.create_index("invoice_no")
txn_col.create_index("customer_id")
cust_col.create_index("customer_id")

# ============================================================
# Helpers
# ============================================================
def strip_ids(docs):
    """Remove _id field for fair inserts."""
    return [{k: v for k, v in doc.items() if k != "_id"} for doc in copy.deepcopy(docs)]

def avg_exec_time(func, docs=None, runs=5, **kwargs):
    """Run a function a few times and return average time taken."""
    total = 0
    for _ in range(runs):
        start = time.perf_counter()
        if docs is not None:
            func(strip_ids(docs), **kwargs)
        else:
            func(**kwargs)
        total += time.perf_counter() - start
    return total / runs

def build_txn_docs(df):
    docs = []
    grouped = df.groupby(["InvoiceNo", "CustomerID", "InvoiceDate", "Country"])
    for (inv, cust, date, country), g in grouped:
        items = g[["StockCode", "Description", "Quantity", "UnitPrice"]].to_dict("records")
        total = (g["Quantity"] * g["UnitPrice"]).sum()
        docs.append({
            "invoice_no": str(inv),
            "customer_id": str(int(cust)),
            "invoice_date": pd.to_datetime(date).isoformat(),
            "country": country,
            "total_amount": round(total, 2),
            "items": items
        })
    return docs

def build_cust_docs(df):
    docs = []
    grouped = df.groupby(["CustomerID", "Country"])
    for (cust, country), g in grouped:
        invoices, spent = [], 0
        for (inv, date), sub in g.groupby(["InvoiceNo", "InvoiceDate"]):
            items = sub[["StockCode", "Description", "Quantity", "UnitPrice"]].to_dict("records")
            total = (sub["Quantity"] * sub["UnitPrice"]).sum()
            spent += total
            invoices.append({
                "invoice_no": str(inv),
                "invoice_date": pd.to_datetime(date).isoformat(),
                "total_amount": round(total, 2),
                "items": items
            })
        docs.append({
            "customer_id": str(int(cust)),
            "country": country,
            "total_spent": round(spent, 2),
            "invoices": invoices
        })
    return docs

txn_docs = build_txn_docs(df)
cust_docs = build_cust_docs(df)

res = []   # store timings

# ============================================================
# CREATE
# ============================================================
print("\n--- CREATE ---")
t1 = avg_exec_time(txn_col.insert_many, txn_docs)
t2 = avg_exec_time(cust_col.insert_many, cust_docs)
res.append(("CREATE - Bulk Insert", t1, t2))

t1 = avg_exec_time(lambda: txn_col.insert_one(strip_ids([txn_docs[0]])[0]))
t2 = avg_exec_time(lambda: cust_col.insert_one(strip_ids([cust_docs[0]])[0]))
res.append(("CREATE - Single Insert", t1, t2))

# ============================================================
# READ
# ============================================================
print("\n--- READ ---")
test_cust = cust_docs[0]["customer_id"]

# customer lookup
t1 = avg_exec_time(lambda: list(txn_col.find({"customer_id": test_cust})), runs=10)
t2 = avg_exec_time(cust_col.find_one, runs=10, **{"filter": {"customer_id": test_cust}})
res.append(("READ - Customer Data Lookup", t1, t2))

# total sales by country
t1 = avg_exec_time(lambda: list(txn_col.aggregate([{"$group": {"_id": "$country", "total": {"$sum": "$total_amount"}}}])), runs=10)
t2 = avg_exec_time(lambda: list(cust_col.aggregate([{"$group": {"_id": "$country", "total": {"$sum": "$total_spent"}}}])), runs=10)
res.append(("READ - Sales by Country", t1, t2))

# invoice lookup
test_invoice = txn_docs[0]["invoice_no"]
t1 = avg_exec_time(txn_col.find_one, runs=10, **{"filter": {"invoice_no": test_invoice}})
t2 = avg_exec_time(lambda: list(cust_col.find({"invoices.invoice_no": test_invoice})), runs=10)
res.append(("READ - Invoice Lookup", t1, t2))

# transaction count
t1 = avg_exec_time(lambda: list(txn_col.aggregate([{"$group": {"_id": "$customer_id", "cnt": {"$sum": 1}}}])), runs=10)
t2 = avg_exec_time(lambda: list(cust_col.aggregate([{"$project": {"customer_id": 1, "cnt": {"$size": "$invoices"}}}])), runs=10)
res.append(("READ - Transaction Count", t1, t2))

# ============================================================
# UPDATE
# ============================================================
print("\n--- UPDATE ---")
test_country = txn_docs[0]["country"]

t1 = avg_exec_time(txn_col.update_many, runs=5, **{"filter": {"customer_id": test_cust}, "update": {"$inc": {"total_amount": 1}}})
t2 = avg_exec_time(cust_col.update_one, runs=5, **{"filter": {"customer_id": test_cust}, "update": {"$inc": {"total_spent": 1}}})
res.append(("UPDATE - Customer Data", t1, t2))

t1 = avg_exec_time(txn_col.update_many, runs=5, **{"filter": {"country": test_country}, "update": {"$inc": {"total_amount": 1}}})
t2 = avg_exec_time(cust_col.update_many, runs=5, **{"filter": {"country": test_country}, "update": {"$inc": {"total_spent": 1}}})
res.append(("UPDATE - By Country", t1, t2))

# ============================================================
# DELETE
# ============================================================
print("\n--- DELETE ---")
t1 = avg_exec_time(txn_col.delete_many, runs=5, **{"filter": {"customer_id": test_cust}})
t2 = avg_exec_time(cust_col.delete_one, runs=5, **{"filter": {"customer_id": test_cust}})
res.append(("DELETE - Customer Data", t1, t2))

t1 = avg_exec_time(txn_col.delete_many, runs=5, **{"filter": {"country": test_country}})
t2 = avg_exec_time(cust_col.delete_many, runs=5, **{"filter": {"country": test_country}})
res.append(("DELETE - By Country", t1, t2))

# ============================================================
# RESULTS
# ============================================================
print("\n============================ RESULTS ============================")
print("Operation".ljust(35), "Txn-Centric".rjust(15), "Cust-Centric".rjust(15))
print("-" * 65)
for op, a, b in res:
    print(f"{op:<35} {a:>12.6f} {b:>12.6f}")

# ============================================================
# PLOT
# ============================================================
ops = [r[0] for r in res]
txn_times = [r[1] for r in res]
cust_times = [r[2] for r in res]

x = np.arange(len(ops))
width = 0.35

plt.figure(figsize=(14, 6))
plt.bar(x - width/2, txn_times, width, label="Txn-Centric")
plt.bar(x + width/2, cust_times, width, label="Cust-Centric")

plt.xticks(x, ops, rotation=45, ha="right")
plt.ylabel("Avg Time (s)")
plt.title("CRUD Performance: Transaction vs Customer")
plt.legend()
plt.tight_layout()
plt.show()
