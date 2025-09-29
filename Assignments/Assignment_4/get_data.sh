#!/bin/bash

DATA_DIR="data"
mkdir -p $DATA_DIR

URL="https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
EXCEL_FILE="$DATA_DIR/online_Retail.xlsx"

echo "Downloading Online Retail dataset..."
wget -O "$EXCEL_FILE" "$URL"

echo "Converting Excel to CSV..."
uv run convert_data.py

echo "Removing the Excel File"
rm -rf "$EXCEL_FILE"

DB_FILE="question1.db"
sqlite3 $DB_FILE < question1.sql
echo "Database $DB_FILE initialized from SQL file!"
