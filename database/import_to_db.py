"""
One-shot script to:
  1. Connect as postgres superuser
  2. Create the 'user' role and 'churn_db' database if they don't exist
  3. Create all tables via SQLAlchemy models
  4. Import all rows from the Excel file into the customers table (upsert)

Run from project root:
    python database/import_to_db.py
"""

import os
import sys
import pandas as pd
from sqlalchemy import create_engine, text

# ---- Paths ------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
EXCEL_PATH   = os.path.join(PROJECT_ROOT, 'data', 'Telco_customer_churn.xlsx')

ADMIN_URL = "postgresql://postgres:4552@localhost:5432/postgres"
APP_URL   = "postgresql://user:password@localhost:5432/churn_db"

# ---- Step 1: Create role + database -----------------------------------------
print("=" * 60)
print("Step 1: Setting up role and database...")
print("=" * 60)

admin_engine = create_engine(ADMIN_URL, isolation_level="AUTOCOMMIT")

with admin_engine.connect() as conn:
    # Create role 'user' if it doesn't exist
    exists = conn.execute(
        text("SELECT 1 FROM pg_roles WHERE rolname = 'user'")
    ).fetchone()
    if not exists:
        conn.execute(text("CREATE USER \"user\" WITH PASSWORD 'password'"))
        print("  [OK] Created role: user")
    else:
        print("  [--] Role 'user' already exists")

    conn.execute(text("ALTER USER \"user\" CREATEDB"))

    # Create churn_db if it doesn't exist
    exists = conn.execute(
        text("SELECT 1 FROM pg_database WHERE datname = 'churn_db'")
    ).fetchone()
    if not exists:
        conn.execute(text("CREATE DATABASE churn_db OWNER \"user\""))
        print("  [OK] Created database: churn_db")
    else:
        print("  [--] Database 'churn_db' already exists")

    conn.execute(text("GRANT ALL PRIVILEGES ON DATABASE churn_db TO \"user\""))
    print("  [OK] Privileges granted to user")

admin_engine.dispose()

# ---- Step 2: Create tables --------------------------------------------------
print()
print("=" * 60)
print("Step 2: Creating tables...")
print("=" * 60)

sys.path.insert(0, PROJECT_ROOT)
from database.connection import Base
from database.models import Customer

app_engine = create_engine(APP_URL)
Base.metadata.create_all(bind=app_engine)
print("  [OK] Tables created / verified")

# ---- Step 3: Read and clean Excel data --------------------------------------
print()
print("=" * 60)
print("Step 3: Reading Excel file...")
print("=" * 60)

df = pd.read_excel(EXCEL_PATH)
print(f"  [OK] Loaded {len(df):,} rows from {os.path.basename(EXCEL_PATH)}")

# Normalise column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(' ', '_', regex=False)
)

# Rename key columns to match the DB model
rename_map = {
    'customerid':    'customer_id',
    'churn_label':   'churn',
    'tenure_months': 'tenure',
}
df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

# Clean numeric columns
df['total_charges']   = pd.to_numeric(df['total_charges'],   errors='coerce').fillna(0.0)
df['monthly_charges'] = pd.to_numeric(df['monthly_charges'], errors='coerce').fillna(0.0)
df['tenure']          = pd.to_numeric(df['tenure'],          errors='coerce').fillna(0).astype(int)

# senior_citizen: 'Yes'/'No' or 1/0 -> int
if 'senior_citizen' in df.columns:
    if pd.api.types.is_numeric_dtype(df['senior_citizen']):
        df['senior_citizen'] = df['senior_citizen'].fillna(0).astype(int)
    else:
        df['senior_citizen'] = df['senior_citizen'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

# churn: keep as 'Yes'/'No' string
if 'churn' in df.columns and pd.api.types.is_numeric_dtype(df['churn']):
    df['churn'] = df['churn'].map({1: 'Yes', 0: 'No'})

print("  [OK] Data cleaned and normalised")

# ---- Step 4: Bulk upsert into PostgreSQL ------------------------------------
print()
print("=" * 60)
print("Step 4: Importing data into PostgreSQL...")
print("=" * 60)

from sqlalchemy.dialects.postgresql import insert as pg_insert

MODEL_COLS = [
    'customer_id', 'gender', 'senior_citizen', 'partner', 'dependents',
    'tenure', 'phone_service', 'multiple_lines', 'internet_service',
    'online_security', 'online_backup', 'device_protection', 'tech_support',
    'streaming_tv', 'streaming_movies', 'contract', 'paperless_billing',
    'payment_method', 'monthly_charges', 'total_charges', 'churn',
]

records = []
for _, row in df.iterrows():
    record = {}
    for col in MODEL_COLS:
        val = row.get(col, None)
        # Convert pandas NA to None
        try:
            if pd.isna(val):
                val = None
        except (TypeError, ValueError):
            pass
        record[col] = val
    records.append(record)

BATCH = 500
with app_engine.connect() as conn:
    for i in range(0, len(records), BATCH):
        batch = records[i : i + BATCH]
        stmt = pg_insert(Customer.__table__).values(batch)
        stmt = stmt.on_conflict_do_update(
            index_elements=['customer_id'],
            set_={col: stmt.excluded[col] for col in MODEL_COLS if col != 'customer_id'}
        )
        conn.execute(stmt)
        conn.commit()
        done = min(i + BATCH, len(records))
        print(f"  Upserted rows {i+1:,} - {done:,} / {len(records):,}", end='\r')

print()
print(f"  [OK] All {len(records):,} rows upserted successfully")

# ---- Step 5: Verify ---------------------------------------------------------
print()
print("=" * 60)
print("Step 5: Verifying...")
print("=" * 60)

with app_engine.connect() as conn:
    total  = conn.execute(text("SELECT COUNT(*) FROM customers")).scalar()
    churns = conn.execute(text("SELECT COUNT(*) FROM customers WHERE churn = 'Yes'")).scalar()
    rate   = churns / total * 100

print(f"  Total rows in DB   : {total:,}")
print(f"  Churned customers  : {churns:,}")
print(f"  Overall churn rate : {rate:.1f}%")
print()
print("Import complete!")
print(f"App connection string: {APP_URL}")

# ---- Step 6: Update connection.py with working DATABASE_URL -----------------
conn_file = os.path.join(PROJECT_ROOT, 'database', 'connection.py')
with open(conn_file) as f:
    content = f.read()

old = 'postgresql://user:password@localhost:5432/churn_db'
if old in content:
    print()
    print("  [OK] connection.py already points to churn_db - no change needed")
else:
    print()
    print("  [INFO] connection.py DATABASE_URL already customised")
