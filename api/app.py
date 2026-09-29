from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from api.database import get_db
from api.models import Transaction as TransactionModel

from pydantic import BaseModel
import xgboost as xgb
import json
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# 1. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="API for fraud probability and risk scoring",
    version="1.0.0"
)


# ============================================================
# 2. CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 3. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "xgboost_model.json"
THRESHOLD_PATH = BASE_DIR / "models" / "threshold.json"


# ============================================================
# 4. LOAD XGBOOST MODEL
# ============================================================

model = xgb.XGBClassifier()

model.load_model(str(MODEL_PATH))


# ============================================================
# 5. LOAD OPTIMIZED THRESHOLD
# ============================================================

with open(THRESHOLD_PATH, "r") as f:
    threshold_data = json.load(f)

threshold = threshold_data["threshold"]


# ============================================================
# 6. INPUT DATA STRUCTURE
# ============================================================

class Transaction(BaseModel):

    Time: float

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float


# ============================================================
# 7. FEATURE ENGINEERING
# ============================================================

def create_engineered_features(transaction: Transaction):

    # Convert Pydantic object to dictionary
    data = transaction.model_dump()

    # Create DataFrame
    df = pd.DataFrame([data])

    # Transaction Hour
    df["TransactionHour"] = (
        (df["Time"] // 3600) % 24
    )

    # HourSin
    df["HourSin"] = np.sin(
        2 * np.pi * df["TransactionHour"] / 24
    )

    # HourCos
    df["HourCos"] = np.cos(
        2 * np.pi * df["TransactionHour"] / 24
    )

    # AmountLog
    df["AmountLog"] = np.log1p(df["Amount"])

    # AmountBucket
    df["AmountBucket"] = pd.cut(
        df["Amount"],
        bins=[
            -np.inf,
            10,
            50,
            100,
            500,
            1000,
            np.inf
        ],
        labels=False
    )

    # Convert bucket to integer
    df["AmountBucket"] = df["AmountBucket"].astype(int)

    return df


# ============================================================
# 8. HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Fraud Detection API is running"
    }


# ============================================================
# 9. PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(
    transaction: Transaction,
    db: Session = Depends(get_db)
):

    try:

        # ----------------------------------------------------
        # Step 1: Feature Engineering
        # ----------------------------------------------------

        df = create_engineered_features(transaction)


        # ----------------------------------------------------
        # Step 2: Select Model Features
        # ----------------------------------------------------

        feature_columns = [

            "Time",

            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
            "V7",
            "V8",
            "V9",
            "V10",
            "V11",
            "V12",
            "V13",
            "V14",
            "V15",
            "V16",
            "V17",
            "V18",
            "V19",
            "V20",
            "V21",
            "V22",
            "V23",
            "V24",
            "V25",
            "V26",
            "V27",
            "V28",

            "Amount",

            "TransactionHour",
            "HourSin",
            "HourCos",
            "AmountLog",
            "AmountBucket"
        ]


        # ----------------------------------------------------
        # Step 3: Create Final Model Input
        # ----------------------------------------------------

        X = df[feature_columns]


        # ----------------------------------------------------
        # DEBUG INFORMATION
        # ----------------------------------------------------

        print("\n========== FEATURE ENGINEERING ==========")

        print(
            "TransactionHour:",
            df["TransactionHour"].iloc[0]
        )

        print(
            "HourSin:",
            df["HourSin"].iloc[0]
        )

        print(
            "HourCos:",
            df["HourCos"].iloc[0]
        )

        print(
            "AmountLog:",
            df["AmountLog"].iloc[0]
        )

        print(
            "AmountBucket:",
            df["AmountBucket"].iloc[0]
        )

        print(
            "\nNumber of features:",
            X.shape[1]
        )

        print("\nFeature names:")

        print(
            X.columns.tolist()
        )

        print(
            "==========================================\n"
        )


        # ----------------------------------------------------
        # Step 4: Verify Feature Count
        # ----------------------------------------------------

        if X.shape[1] != 35:

            raise ValueError(
                f"Expected 35 features, "
                f"but received {X.shape[1]}"
            )


        # ----------------------------------------------------
        # Step 5: Get Fraud Probability
        # ----------------------------------------------------

        fraud_probability = model.predict_proba(X)[0][1]


        # ----------------------------------------------------
        # Step 6: Apply Optimized Threshold
        # ----------------------------------------------------

        if fraud_probability >= threshold:

            prediction = 1
            risk_level = "High"

        else:

            prediction = 0
            risk_level = "Low"


        # ----------------------------------------------------
        # Step 7: Create Database Record
        # ----------------------------------------------------

        db_transaction = TransactionModel(

            transaction_time=transaction.Time,

            amount=transaction.Amount,

            fraud_probability=float(
                fraud_probability
            ),

            threshold=float(
                threshold
            ),

            prediction=int(
                prediction
            ),

            risk_level=risk_level,

            model_version="XGBoost-v1"
        )


        # ----------------------------------------------------
        # Step 8: Save to PostgreSQL
        # ----------------------------------------------------

        db.add(db_transaction)

        db.commit()

        db.refresh(db_transaction)


        # ----------------------------------------------------
        # Step 9: Return Prediction Result
        # ----------------------------------------------------

        return {

            "transaction_id": db_transaction.id,

            "fraud_probability": round(
                float(fraud_probability),
                4
            ),

            "threshold": round(
                float(threshold),
                4
            ),

            "prediction": prediction,

            "risk_level": risk_level,

            "features_used": X.shape[1],

            "database_saved": True
        }


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# 10. GET ALL TRANSACTIONS
# ============================================================

@app.get("/transactions")
def get_transactions(
    db: Session = Depends(get_db)
):

    transactions = (
        db.query(TransactionModel)
        .order_by(
            TransactionModel.created_at.desc()
        )
        .all()
    )

    return transactions


# ============================================================
# 11. GET FRAUD TRANSACTIONS
# ============================================================

@app.get("/transactions/fraud")
def get_fraud_transactions(
    db: Session = Depends(get_db)
):

    fraud_transactions = (
        db.query(TransactionModel)
        .filter(
            TransactionModel.prediction == 1
        )
        .order_by(
            TransactionModel.created_at.desc()
        )
        .all()
    )

    return fraud_transactions


# ============================================================
# 12. GET FRAUD STATISTICS
# ============================================================

@app.get("/statistics")
def get_statistics(
    db: Session = Depends(get_db)
):

    total_transactions = (
        db.query(TransactionModel).count()
    )


    fraud_transactions = (
        db.query(TransactionModel)
        .filter(
            TransactionModel.prediction == 1
        )
        .count()
    )


    legitimate_transactions = (
        db.query(TransactionModel)
        .filter(
            TransactionModel.prediction == 0
        )
        .count()
    )


    # Calculate fraud rate

    if total_transactions > 0:

        fraud_rate = (
            fraud_transactions /
            total_transactions
        )

    else:

        fraud_rate = 0


    return {

        "total_transactions":
            total_transactions,

        "fraud_transactions":
            fraud_transactions,

        "legitimate_transactions":
            legitimate_transactions,

        "fraud_rate":
            round(fraud_rate, 4)
    }