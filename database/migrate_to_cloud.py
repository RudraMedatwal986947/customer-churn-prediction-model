"""
database/migrate_to_cloud.py
Automated migration script to provision and seed tables to a free Cloud PostgreSQL instance
(e.g., Neon Serverless Postgres, Supabase, Render, or AWS RDS).

Usage:
    python database/migrate_to_cloud.py --url "postgresql://username:password@ep-xyz.neon.tech/neondb?sslmode=require"
    or set the environment variable:
    set CLOUD_DATABASE_URL="postgresql://username:password@ep-xyz.neon.tech/neondb?sslmode=require"
    python database/migrate_to_cloud.py
"""

import os
import sys
import argparse
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
EXCEL_PATH = os.path.join(PROJECT_ROOT, 'data', 'Telco_customer_churn.xlsx')

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from database.connection import Base
from database.models import Customer, PredictionLog, CustomCustomer


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Migrate customer churn schema and records to a free cloud PostgreSQL database."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=os.getenv("CLOUD_DATABASE_URL") or os.getenv("DATABASE_URL"),
        help="PostgreSQL connection string for your cloud database."
    )
    return parser.parse_args()


def migrate_to_cloud(db_url: str):
    # Normalize postgres:// to postgresql://
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    print("=" * 70)
    print("Step 1: Connecting to Cloud PostgreSQL instance...")
    print("=" * 70)

    try:
        cloud_engine = create_engine(
            db_url,
            connect_args={"connect_timeout": 10},
            pool_pre_ping=True
        )
        with cloud_engine.connect() as conn:
            version = conn.execute(text("SELECT version();")).scalar()
            print("Successfully connected!")
            print(f"Remote Server Info: {version[:80]}...")
    except Exception as e:
        print(f"Failed to connect to the cloud database: {e}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("Step 2: Creating database schema and tables...")
    print("=" * 70)
    Base.metadata.create_all(bind=cloud_engine)
    print("Tables created/verified: customers, custom_customers, prediction_logs")

    print("\n" + "=" * 70)
    print("Step 3: Loading and cleaning local Excel dataset...")
    print("=" * 70)
    if not os.path.exists(EXCEL_PATH):
        print(f"Error: Dataset not found at {EXCEL_PATH}")
        sys.exit(1)

    df = pd.read_excel(EXCEL_PATH)
    print(f"Loaded {len(df):,} records from {os.path.basename(EXCEL_PATH)}")

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_', regex=False)
    )

    rename_map = {
        'customerid':    'customer_id',
        'churn_label':   'churn',
        'tenure_months': 'tenure',
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    df['total_charges'] = pd.to_numeric(df.get('total_charges', 0), errors='coerce').fillna(0.0)
    df['monthly_charges'] = pd.to_numeric(df.get('monthly_charges', 0), errors='coerce').fillna(0.0)
    df['tenure'] = pd.to_numeric(df.get('tenure', 0), errors='coerce').fillna(0).astype(int)

    if 'senior_citizen' in df.columns:
        if pd.api.types.is_numeric_dtype(df['senior_citizen']):
            df['senior_citizen'] = df['senior_citizen'].fillna(0).astype(int)
        else:
            df['senior_citizen'] = df['senior_citizen'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    if 'churn' in df.columns and pd.api.types.is_numeric_dtype(df['churn']):
        df['churn'] = df['churn'].map({1: 'Yes', 0: 'No'})

    print("Data cleaning and normalization complete.")

    print("\n" + "=" * 70)
    print("Step 4: Bulk uploading records to Cloud PostgreSQL...")
    print("=" * 70)

    model_cols = [
        'customer_id', 'gender', 'senior_citizen', 'partner', 'dependents',
        'tenure', 'phone_service', 'multiple_lines', 'internet_service',
        'online_security', 'online_backup', 'device_protection', 'tech_support',
        'streaming_tv', 'streaming_movies', 'contract', 'paperless_billing',
        'payment_method', 'monthly_charges', 'total_charges', 'churn',
    ]

    records = []
    for _, row in df.iterrows():
        rec = {}
        for col in model_cols:
            val = row.get(col, None)
            try:
                if pd.isna(val):
                    val = None
            except (TypeError, ValueError):
                pass
            rec[col] = val
        records.append(rec)

    batch_size = 150
    import time
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        stmt = pg_insert(Customer.__table__).values(batch)
        stmt = stmt.on_conflict_do_update(
            index_elements=['customer_id'],
            set_={col: stmt.excluded[col] for col in model_cols if col != 'customer_id'}
        )
        for attempt in range(4):
            try:
                with cloud_engine.connect() as conn:
                    conn.execute(stmt)
                    conn.commit()
                break
            except Exception as batch_err:
                if attempt == 3:
                    raise batch_err
                time.sleep(1.5 * (attempt + 1))

        processed = min(i + batch_size, len(records))
        print(f"Uploaded {processed:,} / {len(records):,} customer records...", end='\r')

    print(f"\nAll {len(records):,} customer records successfully migrated!")

    print("\n" + "=" * 70)
    print("Step 5: Validating remote cloud data...")
    print("=" * 70)
    with cloud_engine.connect() as conn:
        total_count = conn.execute(text("SELECT COUNT(*) FROM customers;")).scalar()
        churn_count = conn.execute(text("SELECT COUNT(*) FROM customers WHERE churn = 'Yes';")).scalar()
        print(f"Total customers in Cloud PostgreSQL: {total_count:,}")
        print(f"Churned customer count: {churn_count:,} ({(churn_count / total_count * 100):.1f}%)")

    cloud_engine.dispose()

    print("\n" + "=" * 70)
    print("Migration Successful!")
    print("=" * 70)
    print("Next step: To connect your live Streamlit Cloud app to this database:")
    print("1. Go to https://share.streamlit.io and open your deployed app.")
    print("2. Click Settings -> Secrets.")
    print("3. Paste the following configuration:")
    print("----------------------------------------------------------------------")
    print(f'DATABASE_URL = "{db_url}"')
    print("----------------------------------------------------------------------")
    print("4. Click Save. Streamlit will restart and instantly query your cloud database!")


def main():
    args = parse_arguments()
    db_url = args.url
    if not db_url:
        print("=" * 70)
        print("Cloud PostgreSQL Migration Tool")
        print("=" * 70)
        print("No DATABASE_URL provided via argument or environment variable.")
        db_url = input("Please enter your Cloud PostgreSQL Connection URI: ").strip()

    if not db_url:
        print("Error: No connection string provided. Aborting migration.")
        sys.exit(1)

    migrate_to_cloud(db_url)


if __name__ == "__main__":
    main()
