from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.sql import func

from api.database import Base


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)

    transaction_time = Column(Float)
    amount = Column(Float, nullable=False)

    fraud_probability = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)

    prediction = Column(Integer, nullable=False)
    risk_level = Column(String(20), nullable=False)

    model_version = Column(
        String(50),
        default="XGBoost-v1"
    )

    created_at = Column(
        DateTime,
        server_default=func.now()
    )