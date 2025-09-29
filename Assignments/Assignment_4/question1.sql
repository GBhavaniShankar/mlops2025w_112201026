-- Enable foreign key support
PRAGMA foreign_keys = ON;

-- Create Product Table
CREATE TABLE IF NOT EXISTS Product (
    StockCode TEXT PRIMARY KEY,
    Description TEXT,
    UnitPrice REAL NOT NULL CHECK(UnitPrice >= 0)
);

-- Create Invoice Table
CREATE TABLE IF NOT EXISTS Invoice (
    InvoiceNo TEXT NOT NULL,
    StockCode TEXT NOT NULL,
    Quantity INTEGER NOT NULL,
    InvoiceDate TEXT NOT NULL,
    CustomerID TEXT,
    Country TEXT,
    FOREIGN KEY(StockCode) REFERENCES Product(StockCode)
);
