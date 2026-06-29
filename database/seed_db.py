import pandas as pd
from database.connection import engine, Base, SessionLocal
from database.models import Customer
import os
import sys

def seed_database(filepath="data/Telco_customer_churn.xlsx"):
    print("Creating tables...")
    Base.metadata.create_all(bind=engine)
    
    print(f"Reading dataset from {filepath}...")
    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
        
    db = SessionLocal()
    
    print("Seeding database...")
    count = 0
    for _, row in df.iterrows():
        # Handle empty Total Charges
        total_charges = str(row['Total Charges']).strip()
        if not total_charges:
            total_charges = 0.0
        else:
            try:
                total_charges = float(total_charges)
            except ValueError:
                total_charges = 0.0
                
        # Handle Senior Citizen
        senior = row['Senior Citizen']
        if isinstance(senior, str):
            senior = 1 if senior.lower() == 'yes' else 0
        
        customer = Customer(
            customer_id=str(row['CustomerID']),
            gender=str(row['Gender']),
            senior_citizen=int(senior),
            partner=str(row['Partner']),
            dependents=str(row['Dependents']),
            tenure=int(row['Tenure Months']),
            phone_service=str(row['Phone Service']),
            multiple_lines=str(row['Multiple Lines']),
            internet_service=str(row['Internet Service']),
            online_security=str(row['Online Security']),
            online_backup=str(row['Online Backup']),
            device_protection=str(row['Device Protection']),
            tech_support=str(row['Tech Support']),
            streaming_tv=str(row['Streaming TV']),
            streaming_movies=str(row['Streaming Movies']),
            contract=str(row['Contract']),
            paperless_billing=str(row['Paperless Billing']),
            payment_method=str(row['Payment Method']),
            monthly_charges=float(row['Monthly Charges']),
            total_charges=float(total_charges),
            churn=str(row['Churn Label'])
        )
        
        # Merge to avoid duplicates
        existing = db.query(Customer).filter_by(customer_id=customer.customer_id).first()
        if not existing:
            db.add(customer)
            count += 1
            
    db.commit()
    db.close()
    print(f"Successfully inserted {count} records.")

if __name__ == "__main__":
    seed_database()