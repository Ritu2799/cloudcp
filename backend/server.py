from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import joblib
import pandas as pd
import numpy as np
import requests
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI(title="AI Predictive Autoscaling System")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# LOAD ML MODELS
# ============================================================================

MODELS = {}
FEATURE_COLUMNS = None

try:
    # Model files in backend directory
    import pickle
    catboost_path = ROOT_DIR / 'catboost_hourly_model.pkl'
    lightgbm_path = ROOT_DIR / 'lightgbm_hourly_model.pkl'
    xgboost_path = ROOT_DIR / 'xgboost_hourly_model.pkl'
    feature_cols_path = ROOT_DIR / 'feature_columns_hourly.pkl'
    
    # Load models with encoding for compatibility
    with open(catboost_path, 'rb') as f:
        MODELS['catboost'] = pickle.load(f, encoding='latin1')
    
    MODELS['lightgbm'] = joblib.load(lightgbm_path)
    MODELS['xgboost'] = joblib.load(xgboost_path)
    FEATURE_COLUMNS = joblib.load(feature_cols_path)
    
    logger.info(f"✅ Loaded 3 ML models successfully")
    logger.info(f"✅ Feature columns: {len(FEATURE_COLUMNS)}")
except Exception as e:
    logger.error(f"❌ Error loading models: {e}")
    MODELS = None
    FEATURE_COLUMNS = None

# ============================================================================
# CALENDARIFIC API INTEGRATION
# ============================================================================

CALENDARIFIC_API_KEY = os.environ.get('CALENDARIFIC_API_KEY', '')
CALENDARIFIC_BASE_URL = 'https://calendarific.com/api/v2'

# Hardcoded major festivals for 2024-2026 (fallback when API fails)
HARDCODED_FESTIVALS = {
    # 2024
    '2024-01-26': 'Republic Day',
    '2024-03-25': 'Holi',
    '2024-04-17': 'Ram Navami',
    '2024-08-15': 'Independence Day',
    '2024-10-12': 'Dussehra',
    '2024-11-01': 'Diwali',
    '2024-11-02': 'Diwali (Day 2)',
    '2024-12-25': 'Christmas',
    
    # 2025
    '2025-01-26': 'Republic Day',
    '2025-03-14': 'Holi',
    '2025-04-06': 'Ram Navami',
    '2025-08-15': 'Independence Day',
    '2025-10-02': 'Dussehra',
    '2025-10-20': 'Diwali',
    '2025-10-21': 'Diwali (Day 2)',
    '2025-12-25': 'Christmas',
    '2025-12-31': 'New Year Eve',
    
    # 2026
    '2026-01-26': 'Republic Day',
    '2026-03-03': 'Holi',
    '2026-03-27': 'Ram Navami',
    '2026-08-15': 'Independence Day',
    '2026-10-21': 'Dussehra',
    '2026-11-08': 'Diwali',
    '2026-11-09': 'Diwali (Day 2)',
    '2026-12-25': 'Christmas',
    '2026-12-31': 'New Year Eve',
}

def check_festival_calendarific(date_str: str, country: str = 'IN') -> Dict[str, Any]:
    """Check if date is a festival using Calendarific API with hardcoded fallback"""
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        year = date_obj.year
        month = date_obj.month
        day = date_obj.day
        
        # First try API if key is available
        if CALENDARIFIC_API_KEY:
            url = f"{CALENDARIFIC_BASE_URL}/holidays"
            params = {
                'api_key': CALENDARIFIC_API_KEY,
                'country': country,
                'year': year,
                'month': month,
                'day': day
            }
            
            try:
                response = requests.get(url, params=params, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check if API returned valid data
                    if data.get('meta', {}).get('code') == 200:
                        holidays = data.get('response', {}).get('holidays', [])
                        
                        if holidays:
                            festival_names = [h['name'] for h in holidays]
                            return {
                                'is_festival': 1,
                                'festival_name': festival_names[0],
                                'all_festivals': festival_names
                            }
            except Exception as api_error:
                logger.warning(f"Calendarific API error: {api_error}, using fallback")
        
        # Fallback to hardcoded festivals
        if date_str in HARDCODED_FESTIVALS:
            festival_name = HARDCODED_FESTIVALS[date_str]
            return {
                'is_festival': 1,
                'festival_name': festival_name,
                'all_festivals': [festival_name]
            }
        
        return {'is_festival': 0, 'festival_name': 'None', 'all_festivals': []}
        
    except Exception as e:
        logger.warning(f"Calendarific API error: {e}")
        return {'is_festival': 0, 'festival_name': 'None', 'all_festivals': []}

# ============================================================================
# AWS AUTO SCALING INTEGRATION
# ============================================================================

def get_aws_autoscaling_client():
    """Get AWS Auto Scaling client"""
    aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID', '')
    aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY', '')
    aws_region = os.environ.get('AWS_REGION', 'us-east-1')
    
    if aws_access_key and aws_secret_key:
        return boto3.client(
            'autoscaling',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
    return None

def scale_ec2_instances(predicted_load: float, asg_name: str) -> Dict[str, Any]:
    """Scale EC2 Auto Scaling Group based on predicted load"""
    try:
        client = get_aws_autoscaling_client()
        
        if not client:
            # Mock mode
            logger.info(f"[MOCK] AWS credentials not provided")
            
            # Calculate required instances
            if predicted_load > 5000:
                desired = 10
            elif predicted_load > 3000:
                desired = 5
            elif predicted_load > 1500:
                desired = 3
            else:
                desired = 2
            
            return {
                'success': True,
                'mode': 'mock',
                'predicted_load': predicted_load,
                'desired_capacity': desired,
                'message': f'[MOCK] Would scale to {desired} instances for load {predicted_load:.0f}'
            }
        
        # Real AWS scaling
        if predicted_load > 5000:
            desired = 10
            max_size = 15
        elif predicted_load > 3000:
            desired = 5
            max_size = 10
        elif predicted_load > 1500:
            desired = 3
            max_size = 5
        else:
            desired = 2
            max_size = 3
        
        response = client.set_desired_capacity(
            AutoScalingGroupName=asg_name,
            DesiredCapacity=desired,
            HonorCooldown=True
        )
        
        # Update max size if needed
        client.update_auto_scaling_group(
            AutoScalingGroupName=asg_name,
            MaxSize=max_size
        )
        
        return {
            'success': True,
            'mode': 'real',
            'predicted_load': predicted_load,
            'desired_capacity': desired,
            'max_size': max_size,
            'asg_name': asg_name,
            'message': f'Scaled {asg_name} to {desired} instances'
        }
        
    except (ClientError, NoCredentialsError) as e:
        logger.error(f"AWS scaling error: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': 'Failed to scale AWS instances'
        }

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def prepare_features_for_hour(timestamp: datetime, festival_info: Dict) -> pd.DataFrame:
    """Prepare features for a single hour prediction"""
    
    hour = timestamp.hour
    day_of_week = timestamp.weekday()
    month = timestamp.month
    day = timestamp.day
    year = timestamp.year
    week_of_year = timestamp.isocalendar()[1]
    quarter = (month - 1) // 3 + 1
    day_of_year = timestamp.timetuple().tm_yday
    
    # Cyclical encoding
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7)
    
    # Boolean features
    is_weekend = 1 if day_of_week >= 5 else 0
    is_business_hours = 1 if 9 <= hour <= 18 else 0
    is_peak_hours = 1 if hour in [12, 13, 19, 20, 21] else 0
    is_night = 1 if hour < 6 or hour > 22 else 0
    
    # Festival info
    is_festival = festival_info.get('is_festival', 0)
    festival_name = festival_info.get('festival_name', 'None')
    
    # Campaign (assume no campaign for future predictions)
    is_campaign = 0
    
    # Mock historical data (in production, fetch from database)
    traffic_lag_1h = 1200.0
    traffic_lag_24h = 1500.0
    traffic_lag_168h = 1400.0
    traffic_rolling_mean_24h = 1300.0
    traffic_rolling_std_24h = 200.0
    traffic_rolling_max_24h = 2000.0
    
    # Mock system metrics
    cpu_usage = 45.0
    memory_usage = 60.0
    response_time = 150.0
    error_rate = 0.5
    
    features = {
        'hour': hour,
        'day_of_week': day_of_week,
        'month': month,
        'day': day,
        'year': year,
        'week_of_year': week_of_year,
        'quarter': quarter,
        'day_of_year': day_of_year,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'dow_sin': dow_sin,
        'dow_cos': dow_cos,
        'is_weekend': is_weekend,
        'is_business_hours': is_business_hours,
        'is_peak_hours': is_peak_hours,
        'is_night': is_night,
        'is_festival': is_festival,
        'is_campaign': is_campaign,
        'festival_name': festival_name,
        'traffic_lag_1h': traffic_lag_1h,
        'traffic_lag_24h': traffic_lag_24h,
        'traffic_lag_168h': traffic_lag_168h,
        'traffic_rolling_mean_24h': traffic_rolling_mean_24h,
        'traffic_rolling_std_24h': traffic_rolling_std_24h,
        'traffic_rolling_max_24h': traffic_rolling_max_24h,
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,
        'response_time': response_time,
        'error_rate': error_rate
    }
    
    return pd.DataFrame([features])

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_traffic(timestamp: datetime, model_name: str = 'catboost') -> Dict[str, Any]:
    """Predict traffic for a given timestamp using specified model"""
    
    if not MODELS or not FEATURE_COLUMNS:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    if model_name not in MODELS:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not found")
    
    # Check festival
    date_str = timestamp.strftime('%Y-%m-%d')
    festival_info = check_festival_calendarific(date_str)
    
    # Prepare features
    features_df = prepare_features_for_hour(timestamp, festival_info)
    
    # Ensure columns match training
    features_df = features_df[FEATURE_COLUMNS]
    
    # Predict
    model = MODELS[model_name]
    
    if model_name == 'catboost':
        prediction = model.predict(features_df)[0]
    elif model_name == 'lightgbm':
        prediction = model.predict(features_df, num_iteration=model.best_iteration)[0]
    elif model_name == 'xgboost':
        import xgboost as xgb
        from sklearn.preprocessing import LabelEncoder
        
        # Encode festival_name
        features_encoded = features_df.copy()
        if 'festival_name' in features_encoded.columns:
            le = LabelEncoder()
            le.fit(['None', 'Diwali', 'Holi', 'Christmas', 'Independence Day', 'Republic Day', 'Ram Navami', 'Diwali Weekend'])
            try:
                features_encoded['festival_name'] = le.transform(features_encoded['festival_name'].astype(str))
            except:
                features_encoded['festival_name'] = 0
        
        dmatrix = xgb.DMatrix(features_encoded)
        prediction = model.predict(dmatrix)[0]
    
    # Ensure positive prediction
    prediction = max(prediction, 50.0)
    
    return {
        'timestamp': timestamp.isoformat(),
        'hour': timestamp.hour,
        'predicted_load': float(prediction),
        'is_festival': festival_info['is_festival'],
        'festival_name': festival_info['festival_name'],
        'model': model_name
    }

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class PredictionRequest(BaseModel):
    start_time: str  # ISO format datetime
    hours: int = 24
    model_name: str = 'catboost'

class PredictionResponse(BaseModel):
    timestamp: str
    hour: int
    predicted_load: float
    is_festival: int
    festival_name: str
    model: str

class ScalingRequest(BaseModel):
    predicted_load: float
    asg_name: str = 'my-asg'

class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

# ============================================================================
# API ROUTES
# ============================================================================

@api_router.get("/")
async def root():
    return {
        "message": "AI Predictive Autoscaling System",
        "models_loaded": MODELS is not None,
        "available_models": list(MODELS.keys()) if MODELS else []
    }

@api_router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": MODELS is not None,
        "feature_columns": len(FEATURE_COLUMNS) if FEATURE_COLUMNS else 0
    }

@api_router.post("/predict", response_model=List[PredictionResponse])
async def predict_endpoint(request: PredictionRequest):
    """Predict traffic for next N hours"""
    try:
        start_time = datetime.fromisoformat(request.start_time)
        predictions = []
        
        for i in range(request.hours):
            timestamp = start_time + timedelta(hours=i)
            pred = predict_traffic(timestamp, request.model_name)
            predictions.append(pred)
        
        return predictions
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/scale")
async def scale_endpoint(request: ScalingRequest):
    """Scale AWS EC2 Auto Scaling Group"""
    try:
        result = scale_ec2_instances(request.predicted_load, request.asg_name)
        return result
    except Exception as e:
        logger.error(f"Scaling error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/festivals/{date}")
async def check_festival(date: str):
    """Check if date is a festival"""
    try:
        festival_info = check_festival_calendarific(date)
        return festival_info
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.get("/models")
async def get_models():
    """Get available models"""
    return {
        "models": list(MODELS.keys()) if MODELS else [],
        "default": "catboost"
    }

@api_router.get("/next-festival")
async def get_next_festival(model_name: str = 'catboost'):
    """Get next upcoming festival with predictions"""
    try:
        today = datetime.now()
        
        # Look for next festival within next 60 days
        for days_ahead in range(1, 61):
            check_date = today + timedelta(days=days_ahead)
            date_str = check_date.strftime('%Y-%m-%d')
            
            festival_info = check_festival_calendarific(date_str)
            
            if festival_info['is_festival'] == 1:
                # Found next festival, get 24h predictions
                predictions = []
                for hour in range(24):
                    timestamp = check_date.replace(hour=hour, minute=0, second=0, microsecond=0)
                    pred = predict_traffic(timestamp, model_name)
                    predictions.append(pred)
                
                # Calculate metrics
                loads = [p['predicted_load'] for p in predictions]
                avg_load = sum(loads) / len(loads)
                peak_load = max(loads)
                
                # Recommended instances based on peak load
                if peak_load > 5000:
                    recommended_instances = 10
                elif peak_load > 3000:
                    recommended_instances = 5
                elif peak_load > 1500:
                    recommended_instances = 3
                else:
                    recommended_instances = 2
                
                return {
                    'festival_name': festival_info['festival_name'],
                    'date': date_str,
                    'days_until': days_ahead,
                    'avg_load': round(avg_load),
                    'peak_load': round(peak_load),
                    'recommended_instances': recommended_instances,
                    'predictions': predictions
                }
        
        # No festival found in next 60 days
        return {
            'festival_name': None,
            'date': None,
            'days_until': None,
            'avg_load': 0,
            'peak_load': 0,
            'recommended_instances': 0,
            'predictions': []
        }
        
    except Exception as e:
        logger.error(f"Next festival error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Original routes
@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.model_dump()
    status_obj = StatusCheck(**status_dict)
    
    doc = status_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    
    _ = await db.status_checks.insert_one(doc)
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find({}, {"_id": 0}).to_list(1000)
    
    for check in status_checks:
        if isinstance(check['timestamp'], str):
            check['timestamp'] = datetime.fromisoformat(check['timestamp'])
    
    return status_checks

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
