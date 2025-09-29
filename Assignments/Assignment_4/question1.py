import sqlite3
import pandas as pd

def connect_db(db_file="question1.db"):
    conn = sqlite3.connect(db_file)
    return conn

def insert_from_dataframe(conn, df: pd.DataFrame):
    products = df[['StockCode', 'Description', 'UnitPrice']].drop_duplicates()

    invoices = df[['InvoiceNo', 'StockCode', 'Quantity', 'InvoiceDate', 'CustomerID', 'Country']]

    cur = conn.cursor()

    for _, row in products.iterrows():
        try:
            cur.execute("""
                INSERT OR IGNORE INTO Product (StockCode, Description, UnitPrice)
                VALUES (?, ?, ?)
            """, (row['StockCode'], row['Description'], row['UnitPrice']))
        except sqlite3.Error as e:
            print("Error inserting product:", e)

    for _, row in invoices.iterrows():
        try:
            cur.execute("""
                INSERT INTO Invoice (InvoiceNo, StockCode, Quantity, InvoiceDate, CustomerID, Country)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (row['InvoiceNo'], row['StockCode'], row['Quantity'],
                  row['InvoiceDate'], row['CustomerID'], row['Country']))
        except sqlite3.Error as e:
            print("Error inserting invoice:", e)
            print(row['Quantity'])

    conn.commit()
    print("Data inserted successfully!")


if __name__ == "__main__":
    conn = connect_db()

    df = pd.read_csv("data/online_retail.csv")
    df = df.iloc[:1000]

    insert_from_dataframe(conn, df)

    conn.close()
