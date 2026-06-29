from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from database.connection import Base
from datetime import datetime

class Customer(Base):
    __tablename__ = "customers"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String, unique=True, index=True, nullable=False)
    
    # Demographics
    gender = Column(String)
    senior_citizen = Column(Integer)
    partner = Column(String)
    dependents = Column(String)
    
    # Account Information
    tenure = Column(Integer)
    contract = Column(String)
    paperless_billing = Column(String)
    payment_method = Column(String)
    
    # Services
    phone_service = Column(String)
    multiple_lines = Column(String)
    internet_service = Column(String)
    online_security = Column(String)
    online_backup = Column(String)
    device_protection = Column(String)
    tech_support = Column(String)
    streaming_tv = Column(String)
    streaming_movies = Column(String)
    
    # Financials
    monthly_charges = Column(Float)
    total_charges = Column(Float)
    
    # Labels
    churn = Column(String)
    
    # Added fields for predictions
    predicted_churn = Column(Float, nullable=True)
    predicted_clv = Column(Float, nullable=True)
    segment = Column(String, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
