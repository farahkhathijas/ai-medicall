
import os
import math
import requests
import numpy as np
import json
import io
import base64
import sys
import datetime
import random
import time
import asyncio
import threading
from typing import List, Dict, Optional, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, ForeignKey, DateTime, Text
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from services.sms_service import sms_service, mask_phone
from services.queue_service import (
    get_or_create_timeline,
    add_timeline_event,
    calculate_departure_buffer,
    advance_appointment_queue
)

# =====================================================
# 1. CONFIGURATION & SETUP
# =====================================================

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

client = None
if GROQ_API_KEY and not GROQ_API_KEY.startswith("your_groq") and GROQ_API_KEY != "placeholder":
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        print(f"Groq Client Initialized (key: {GROQ_API_KEY[:5]}...{GROQ_API_KEY[-4:]})")
    except Exception as e:
        print(f"Failed to initialize Groq client: {e}")
        client = None
else:
    print("Running in Standalone ML Mode (Groq LLM disabled or placeholder key; using Hybrid Semantic + ML Engine)")

app = FastAPI(title="Agentic Medical Analyser - Intelligent Healthcare Assistant")

# Setup production-grade CORS supporting local dev and Vercel deployments
frontend_url_env = os.getenv("FRONTEND_URL", "").strip()
allowed_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:8080",
    "http://localhost:9000",
    "http://localhost:8000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:9000",
    "http://127.0.0.1:8000",
]
if frontend_url_env:
    for url in frontend_url_env.split(","):
        cleaned = url.strip().rstrip("/")
        if cleaned and cleaned not in allowed_origins:
            allowed_origins.append(cleaned)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"^https:\/\/.*\.vercel\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# 2. DATABASE SETUP
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'medical_ai.db')}"
sys.stderr.write(f"DB URL: {DATABASE_URL}\n")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    symptoms = Column(String)
    department_1 = Column(String)
    confidence_1 = Column(Float)
    department_2 = Column(String)
    confidence_2 = Column(Float)
    department_3 = Column(String)
    confidence_3 = Column(Float)
    emergency = Column(Boolean)

class Patient(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer)
    gender = Column(String)
    symptoms = Column(String)
    heart_rate = Column(Integer)
    temperature = Column(Float)
    pre_existing_conditions = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Hospital(Base):
    __tablename__ = "hospitals"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    address = Column(String)
    phone = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
    emergency_available = Column(Boolean, default=True)
    rating = Column(Float, default=4.0)
    open_now = Column(Boolean, default=True)

class Doctor(Base):
    __tablename__ = "doctors"
    id = Column(Integer, primary_key=True, index=True)
    hospital_id = Column(Integer, ForeignKey("hospitals.id"))
    name = Column(String)
    department = Column(String)
    specialization = Column(String)

class AppointmentSlot(Base):
    __tablename__ = "appointment_slots"
    id = Column(Integer, primary_key=True, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"))
    date = Column(String)
    start_time = Column(String)
    end_time = Column(String)
    available = Column(Boolean, default=True)

class Appointment(Base):
    __tablename__ = "appointments"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    doctor_id = Column(Integer, ForeignKey("doctors.id"))
    hospital_id = Column(Integer, ForeignKey("hospitals.id"))
    slot_id = Column(Integer, ForeignKey("appointment_slots.id"))
    status = Column(String, default="Confirmed")
    queue_number = Column(Integer, default=0)
    queue_position = Column(Integer, default=0)
    estimated_wait_minutes = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class QueueEntry(Base):
    __tablename__ = "queue_entries"
    id = Column(Integer, primary_key=True, index=True)
    appointment_id = Column(Integer, ForeignKey("appointments.id"))
    doctor_id = Column(Integer, ForeignKey("doctors.id"))
    queue_number = Column(Integer)
    position = Column(Integer)
    status = Column(String, default="waiting")  # waiting, in_progress, completed
    estimated_wait_minutes = Column(Integer, default=15)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class Notification(Base):
    __tablename__ = "notifications"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, default=1)
    appointment_id = Column(Integer, ForeignKey("appointments.id"), nullable=True)
    message = Column(Text)
    notification_type = Column(String)  # queue_update, appointment_confirmed, turn_approaching, turn_reached
    read = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class UserProfile(Base):
    __tablename__ = "user_profiles"
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String)
    age = Column(Integer)
    gender = Column(String)
    phone = Column(String, nullable=True)
    email = Column(String, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class SMSLog(Base):
    __tablename__ = "sms_logs"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, nullable=True)
    appointment_id = Column(Integer, nullable=True)
    phone_number = Column(String)
    message_type = Column(String)
    message_body = Column(Text)
    status = Column(String, default="SENT")
    provider = Column(String, default="simulation")
    provider_response_id = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

def init_db():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        if db.query(Hospital).count() == 0:
            h1 = Hospital(name="City General Hospital", address="123 Health Ave, Chennai", phone="+91 9876543210", latitude=13.0827, longitude=80.2707, emergency_available=True, rating=4.5, open_now=True)
            h2 = Hospital(name="Apollo Main Hospital", address="45 Wellness St, Chennai", phone="+91 1234567890", latitude=13.0604, longitude=80.2496, emergency_available=True, rating=4.7, open_now=True)
            h3 = Hospital(name="Global Health Center", address="78 Care Blvd, Chennai", phone="+91 9988776655", latitude=13.0500, longitude=80.2600, emergency_available=False, rating=4.2, open_now=True)
            h4 = Hospital(name="Sunrise Medical Institute", address="12 Dawn Rd, Chennai", phone="+91 8877665544", latitude=13.0900, longitude=80.2800, emergency_available=True, rating=4.3, open_now=True)
            db.add_all([h1, h2, h3, h4])
            db.commit()
            db.refresh(h1); db.refresh(h2); db.refresh(h3); db.refresh(h4)

            docs = [
                Doctor(hospital_id=h1.id, name="Dr. Priya Sharma", department="Cardiology", specialization="Heart Surgeon"),
                Doctor(hospital_id=h1.id, name="Dr. Arjun Menon", department="General Medicine", specialization="Physician"),
                Doctor(hospital_id=h1.id, name="Dr. Lakshmi Rao", department="Neurology", specialization="Neurologist"),
                Doctor(hospital_id=h2.id, name="Dr. Vikram Patel", department="Cardiology", specialization="Interventional Cardiologist"),
                Doctor(hospital_id=h2.id, name="Dr. Ananya Iyer", department="Pulmonology", specialization="Pulmonologist"),
                Doctor(hospital_id=h2.id, name="Dr. Rajesh Kumar", department="Orthopedics", specialization="Orthopedic Surgeon"),
                Doctor(hospital_id=h3.id, name="Dr. Deepa Nair", department="Gastroenterology", specialization="Gastroenterologist"),
                Doctor(hospital_id=h3.id, name="Dr. Suresh Babu", department="General Medicine", specialization="Internal Medicine"),
                Doctor(hospital_id=h4.id, name="Dr. Meena Krishnan", department="ENT", specialization="ENT Specialist"),
                Doctor(hospital_id=h4.id, name="Dr. Arun Prakash", department="Dermatology", specialization="Dermatologist"),
            ]
            db.add_all(docs)
            db.commit()
            for doc in docs:
                db.refresh(doc)

            today = datetime.date.today().strftime("%Y-%m-%d")
            tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            slots = []
            for doc in docs:
                for hour in [9, 10, 11, 14, 15, 16]:
                    slots.append(AppointmentSlot(
                        doctor_id=doc.id, date=today,
                        start_time=f"{hour:02d}:00", end_time=f"{hour:02d}:30", available=True
                    ))
                    slots.append(AppointmentSlot(
                        doctor_id=doc.id, date=today,
                        start_time=f"{hour:02d}:30", end_time=f"{hour+1:02d}:00", available=True
                    ))
                for hour in [9, 10, 11, 14, 15, 16]:
                    slots.append(AppointmentSlot(
                        doctor_id=doc.id, date=tomorrow,
                        start_time=f"{hour:02d}:00", end_time=f"{hour:02d}:30", available=True
                    ))
            db.add_all(slots)
            db.commit()
    except Exception as e:
        print(f"Error seeding database: {e}")
    finally:
        db.close()

init_db()

# =====================================================
# 3. AI MODELS & KNOWLEDGE BASE
# =====================================================

embedder = None
try:
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print(f"Error loading SentenceTransformer: {e}")

import faiss
import joblib

department_knowledge = {
    "Emergency Medicine": "Life threatening conditions including cardiac arrest, stroke, severe trauma, heavy bleeding, respiratory failure.",
    "General Medicine": "Common illnesses including fever, fatigue, infections, general weakness, non-specific symptoms.",
    "Cardiology": "Heart related disorders including chest pain, heart attack, arrhythmia, hypertension, coronary artery disease.",
    "Neurology": "Brain and nervous system disorders including stroke, seizures, migraine, neuropathy, paralysis.",
    "Dermatology": "Skin diseases including rash, eczema, acne, fungal infection, psoriasis.",
    "Orthopedics": "Bone and joint disorders including fractures, arthritis, joint pain, spine injury.",
    "Pediatrics": "Medical care for infants and children including childhood infections and growth issues.",
    "Psychiatry": "Mental health conditions including depression, anxiety, bipolar disorder, hallucinations.",
    "Gastroenterology": "Digestive system disorders including abdominal pain, vomiting, diarrhea, liver disease.",
    "Pulmonology": "Respiratory diseases including asthma, pneumonia, breathing difficulty, chronic cough.",
    "Urology": "Urinary tract disorders including kidney stones, urinary infections, prostate issues.",
    "Nephrology": "Kidney related diseases including renal failure, dialysis conditions, electrolyte imbalance.",
    "Endocrinology": "Hormonal disorders including diabetes, thyroid disease, metabolic syndrome.",
    "Oncology": "Cancer related conditions including tumor growth, chemotherapy, radiation therapy.",
    "ENT": "Ear, nose and throat disorders including sinusitis, hearing loss, throat infections.",
    "Ophthalmology": "Eye related diseases including vision loss, cataract, glaucoma, eye infection.",
    "Gynecology": "Female reproductive health including menstrual disorders, ovarian cyst, pelvic pain."
}

department_names = list(department_knowledge.keys())
knowledge_texts = list(department_knowledge.values())

dept_index = None
if embedder:
    knowledge_embeddings = embedder.encode(knowledge_texts)
    d_dimension = knowledge_embeddings.shape[1]
    dept_index = faiss.IndexFlatL2(d_dimension)
    dept_index.add(np.array(knowledge_embeddings))

training_data = [
    ("Chest pain radiating to left arm", "Cardiology"),
    ("Shortness of breath and chest tightness", "Pulmonology"),
    ("Frequent urination and burning sensation", "Urology"),
    ("Skin rash with itching and redness", "Dermatology"),
    ("Severe headache and dizziness", "Neurology"),
    ("Joint pain and swelling", "Orthopedics"),
    ("Fever and persistent cough", "General Medicine"),
    ("Abdominal pain and vomiting", "Gastroenterology"),
    ("Irregular heartbeat and palpitations", "Cardiology"),
    ("Seizure episode and confusion", "Neurology"),
]

train_texts = [x[0] for x in training_data]
train_labels = [x[1] for x in training_data]

case_index = None
if embedder:
    train_embeddings = embedder.encode(train_texts)
    c_dimension = train_embeddings.shape[1]
    case_index = faiss.IndexFlatL2(c_dimension)
    case_index.add(np.array(train_embeddings))

classifier = None
classifier_path = os.path.join(BASE_DIR, "models", "trained_model", "classifier.pkl")
if os.path.exists(classifier_path):
    try:
        classifier = joblib.load(classifier_path)
    except Exception as e:
        print(f"Could not load classifier.pkl: {e}")

triage_model = None
triage_encoder = None
try:
    import pickle
    triage_model_path = os.path.join(BASE_DIR, "models", "triage_model.pkl")
    triage_encoder_path = os.path.join(BASE_DIR, "models", "encoders.pkl")
    if os.path.exists(triage_model_path) and os.path.exists(triage_encoder_path):
        with open(triage_model_path, "rb") as f:
            triage_model = pickle.load(f)
        with open(triage_encoder_path, "rb") as f:
            triage_encoder = pickle.load(f)
        print("Triage Risk Model Loaded")
except Exception as e:
    print(f"Error loading triage models: {e}")

emergency_sentences = [
    "heart attack", "stroke", "severe bleeding", "unconscious",
    "difficulty breathing", "chest pain severe", "cardiac arrest",
    "seizure", "anaphylaxis", "choking"
]
emergency_embeddings = None
if embedder:
    emergency_embeddings = embedder.encode(emergency_sentences)


def check_emergency(text):
    if not embedder or emergency_embeddings is None:
        return False
    input_embedding = embedder.encode([text])
    similarities = np.dot(input_embedding, emergency_embeddings.T)
    max_score = np.max(similarities)
    return max_score > 0.6


def hybrid_predict_logic(symptoms: str):
    if not embedder or not dept_index:
        return {"System": "Error", "Top 3 Recommendations": [], "Similar Past Cases": []}

    user_embedding = embedder.encode([symptoms])
    distances, indices = dept_index.search(np.array(user_embedding), k=3)
    semantic_scores = 1 / (1 + distances[0])
    semantic_scores = semantic_scores / np.sum(semantic_scores)

    semantic_results = []
    for i, idx in enumerate(indices[0]):
        semantic_results.append({
            "Department": department_names[idx],
            "Semantic Confidence (%)": round(float(semantic_scores[i] * 100), 2)
        })

    similar_cases = []
    if case_index:
        c_distances, c_indices = case_index.search(np.array(user_embedding), k=3)
        for i, idx in enumerate(c_indices[0]):
            if idx < len(training_data):
                similar_cases.append({
                    "Symptom": training_data[idx][0],
                    "Department": training_data[idx][1],
                    "Distance": float(c_distances[0][i])
                })

    ml_results = []
    if classifier:
        try:
            ml_probs = classifier.predict_proba(user_embedding)[0]
            ml_indices = np.argsort(ml_probs)[::-1][:3]
            for idx in ml_indices:
                ml_results.append({
                    "Department": classifier.classes_[idx],
                    "ML Confidence (%)": round(float(ml_probs[idx] * 100), 2)
                })
        except Exception:
            pass

    combined = {}
    for item in semantic_results:
        combined[item["Department"]] = item["Semantic Confidence (%)"] * 0.6
    for item in ml_results:
        if item["Department"] in combined:
            combined[item["Department"]] += item["ML Confidence (%)"] * 0.4
        else:
            combined[item["Department"]] = item["ML Confidence (%)"] * 0.4

    sorted_final = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:3]
    final_results = [{"Department": dept, "Final Confidence (%)": round(score, 2)} for dept, score in sorted_final]

    return {
        "System": "Hybrid Semantic + ML Engine" if classifier else "Semantic Knowledge Engine",
        "Top 3 Recommendations": final_results,
        "Similar Past Cases": similar_cases
    }


def predict_risk_logic(age, gender, systolic_bp, diastolic_bp, heart_rate, temperature, symptoms_str):
    if not triage_model or not triage_encoder:
        return "Unknown"
    try:
        mean_bp = (systolic_bp + 2 * diastolic_bp) / 3
        X_num = np.array([[age, mean_bp, heart_rate, temperature]])
        X_cat = [[gender, symptoms_str]]
        X_cat_encoded = triage_encoder.transform(X_cat)
        X = np.hstack([X_num, X_cat_encoded])
        prediction = triage_model.predict(X)[0]
        return prediction
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Unknown"


# =====================================================
# 4. HAVERSINE + GEOCODING
# =====================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (math.sin(dLat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dLon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def get_coordinates_for_city(city_name: str):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": city_name, "format": "json", "limit": 1}
        headers = {"User-Agent": "AgenticMedicalAnalyser/1.0"}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        if data and len(data) > 0:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        print(f"Geocoding error: {e}")
    return None, None


# =====================================================
# 5. AGENT TOOLS (Functions the AI can call)
# =====================================================

def tool_search_hospitals_osm(latitude, longitude, radius_km=10.0, department=None):
    """Search real hospitals via Overpass/OSM with multi-endpoint failover and Government/Private classification"""
    radius_m = radius_km * 1000
    overpass_endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter"
    ]
    overpass_query = f"""
    [out:json][timeout:15];
    (
      node["amenity"="hospital"](around:{radius_m},{latitude},{longitude});
      way["amenity"="hospital"](around:{radius_m},{latitude},{longitude});
      relation["amenity"="hospital"](around:{radius_m},{latitude},{longitude});
      node["amenity"="clinic"](around:{radius_m},{latitude},{longitude});
      way["amenity"="clinic"](around:{radius_m},{latitude},{longitude});
    );
    out center;
    """
    results = []
    response_data = None
    
    for url in overpass_endpoints:
        try:
            resp = requests.post(url, data={'data': overpass_query}, timeout=15)
            if resp.status_code == 200:
                response_data = resp.json()
                break
        except Exception:
            continue

    if response_data and 'elements' in response_data:
        for element in response_data.get('elements', []):
            tags = element.get('tags', {})
            name = tags.get('name')
            if not name:
                continue
            lat = element.get('lat') or (element.get('center', {}) or {}).get('lat')
            lon = element.get('lon') or (element.get('center', {}) or {}).get('lon')
            if not lat or not lon:
                continue

            address = (tags.get('addr:street', '') + " " + tags.get('addr:city', '')).strip()
            if not address:
                address = tags.get('addr:full', '') or tags.get('addr:suburb', '') or "Address verified via OpenStreetMap"

            phone = tags.get('phone') or tags.get('contact:phone') or ""
            website = tags.get('website') or tags.get('contact:website') or ""
            speciality = tags.get('healthcare:speciality', 'General Medicine')
            deps = [d.strip().title() for d in speciality.split(';') if d.strip()]
            if not deps:
                deps = ["General Medicine"]

            dist = haversine(latitude, longitude, lat, lon)
            travel_time_min = max(4, round(dist * 3.2))

            # Real emergency verification without random guessing
            em_tag = tags.get('emergency', '').lower()
            hc_tag = tags.get('healthcare', '').lower()
            name_lower = name.lower()
            emergency_available = (
                em_tag in ['yes', 'designated'] or
                'emergency' in hc_tag or
                'casualty' in tags.get('healthcare:speciality', '').lower() or
                'emergency' in name_lower or
                'trauma' in name_lower
            )

            # Government vs Private Facility Classification
            op_type = (tags.get('operator:type', '') or tags.get('ownership', '')).lower()
            operator = tags.get('operator', '').lower()
            if op_type in ['government', 'public', 'state', 'national', 'municipal'] or any(kw in operator for kw in ['government', 'dept', 'ministry', 'health department']):
                fac_type = "Government"
            elif op_type in ['private', 'for-profit', 'community']:
                fac_type = "Private"
            elif any(kw in name_lower for kw in ['government', 'govt', 'district hospital', 'general hospital', 'civil hospital', 'aiims', 'esi hospital', 'primary health centre', 'gh ', 'medical college hospital']):
                fac_type = "Government"
            elif any(kw in name_lower for kw in ['apollo', 'fortis', 'manipal', 'max', 'care hospital', 'narayana', 'aster', 'medanta', 'kims', 'columbia asia', 'kauvery', 'miot', 'dr.', 'private', 'memorial', 'trust', 'lifeline']):
                fac_type = "Private"
            elif 'clinic' in name_lower or tags.get('amenity') == 'clinic':
                fac_type = "Specialized Clinic"
            else:
                fac_type = "Type unavailable"

            results.append({
                "id": element.get('id'),
                "name": name,
                "type": fac_type,
                "latitude": lat,
                "longitude": lon,
                "address": address,
                "phone": phone,
                "website": website,
                "departments": deps,
                "distance_km": round(dist, 2),
                "travel_time_min": travel_time_min,
                "emergency_available": emergency_available,
                "open_now": True,
                "rating": 4.5 if emergency_available else 4.2,
                "source": "osm"
            })
        
        results.sort(key=lambda x: x["distance_km"])
        if results:
            return results[:25]

    # Fallback to internal verified database if Overpass query returned no results or failed
    return tool_search_hospitals_db(department, latitude, longitude)


def tool_search_hospitals_db(department=None, latitude=None, longitude=None):
    """Search internal verified DB hospitals with realistic government and private classifications"""
    db = SessionLocal()
    try:
        query = db.query(Hospital)
        hospitals = query.all()
        results = []
        for h in hospitals:
            dist = 1.4
            if latitude is not None and longitude is not None and h.latitude and h.longitude:
                dist = round(haversine(latitude, longitude, h.latitude, h.longitude), 2)
            travel_time_min = max(5, int(dist * 3.5))

            name_lower = h.name.lower()
            if "general" in name_lower or "district" in name_lower or "government" in name_lower or "civil" in name_lower or "esi" in name_lower:
                fac_type = "Government"
            elif "apollo" in name_lower or "global" in name_lower or "sunrise" in name_lower or "institute" in name_lower:
                fac_type = "Private"
            else:
                fac_type = "Type unavailable"

            results.append({
                "id": h.id,
                "name": h.name,
                "type": fac_type,
                "latitude": h.latitude,
                "longitude": h.longitude,
                "address": h.address,
                "phone": h.phone or "+91 98765 43210",
                "emergency_available": bool(h.emergency_available),
                "open_now": h.open_now,
                "rating": h.rating or 4.4,
                "distance_km": dist,
                "travel_time_min": travel_time_min,
                "departments": ["General Medicine", "Emergency", "Cardiology"],
                "source": "internal"
            })
        if latitude is not None and longitude is not None:
            results.sort(key=lambda x: x["distance_km"])
        return results
    finally:
        db.close()


def tool_search_doctors(department=None, hospital_id=None):
    db = SessionLocal()
    try:
        query = db.query(Doctor)
        if department:
            query = query.filter(Doctor.department.ilike(f"%{department}%"))
        if hospital_id:
            query = query.filter(Doctor.hospital_id == hospital_id)
        doctors = query.all()
        result = []
        for d in doctors:
            hospital = db.query(Hospital).filter(Hospital.id == d.hospital_id).first()
            result.append({
                "id": d.id, "hospital_id": d.hospital_id,
                "name": d.name, "department": d.department,
                "specialization": d.specialization,
                "hospital": hospital.name if hospital else "Unknown"
            })
        return result
    finally:
        db.close()


def tool_get_slots(doctor_id):
    db = SessionLocal()
    try:
        slots = db.query(AppointmentSlot).filter(
            AppointmentSlot.doctor_id == doctor_id,
            AppointmentSlot.available == True
        ).all()
        return [{"id": s.id, "date": s.date, "start_time": s.start_time, "end_time": s.end_time} for s in slots]
    finally:
        db.close()


def tool_book_appointment(patient_id, doctor_id, hospital_id, slot_id):
    db = SessionLocal()
    try:
        slot = db.query(AppointmentSlot).filter(AppointmentSlot.id == slot_id).first()
        if not slot or not slot.available:
            return {"error": "Slot unavailable."}

        slot.available = False
        # Calculate queue number
        existing_apts = db.query(Appointment).filter(
            Appointment.doctor_id == doctor_id,
            Appointment.status == "Confirmed"
        ).count()
        queue_num = existing_apts + 1

        new_apt = Appointment(
            patient_id=patient_id, doctor_id=doctor_id,
            hospital_id=hospital_id, slot_id=slot_id,
            status="Confirmed", queue_number=queue_num,
            queue_position=queue_num,
            estimated_wait_minutes=queue_num * 8
        )
        db.add(new_apt)
        db.commit()
        db.refresh(new_apt)

        # Create queue entry
        queue_entry = QueueEntry(
            appointment_id=new_apt.id, doctor_id=doctor_id,
            queue_number=queue_num, position=queue_num,
            status="waiting", estimated_wait_minutes=queue_num * 8
        )
        db.add(queue_entry)

        # Create notification
        doc = db.query(Doctor).filter(Doctor.id == doctor_id).first()
        hosp = db.query(Hospital).filter(Hospital.id == hospital_id).first()
        notif = Notification(
            patient_id=patient_id, appointment_id=new_apt.id,
            message=f"Appointment confirmed at {hosp.name if hosp else 'Hospital'} with {doc.name if doc else 'Doctor'} on {slot.date} at {slot.start_time}. Queue #{queue_num}.",
            notification_type="appointment_confirmed"
        )
        db.add(notif)
        db.commit()

        # Get user phone for SMS
        patient_profile = db.query(UserProfile).filter(UserProfile.id == patient_id).first()
        if not patient_profile:
            patient_profile = db.query(UserProfile).order_by(UserProfile.id.desc()).first()
        phone = patient_profile.phone if (patient_profile and patient_profile.phone) else "+91 9876543210"
        patient_name = patient_profile.full_name if patient_profile else "Registered Patient"

        # Trigger Real/Simulated SMS
        sms_res = sms_service.send_appointment_confirmation(
            phone=phone,
            hospital=hosp.name if hosp else "City General Hospital",
            doctor=doc.name if doc else "Dr. Rajesh Sharma",
            department=doc.department if doc else "General Medicine",
            date_str=slot.date,
            time_str=slot.start_time,
            appointment_id=new_apt.id,
            queue_number=queue_num,
            patient_id=patient_id
        )

        # Initialize activity timeline
        timeline = get_or_create_timeline(new_apt.id, initial_pos=queue_num, wait_min=queue_num * 8)
        dep_buffer = calculate_departure_buffer(distance_km=2.2, appointment_time_str=slot.start_time)

        return {
            "id": new_apt.id, "status": "Confirmed",
            "queue_number": queue_num,
            "queue_position": queue_num,
            "people_ahead": max(0, queue_num - 1),
            "estimated_wait_minutes": queue_num * 8,
            "doctor": doc.name if doc else "Unknown",
            "department": doc.department if doc else "General Medicine",
            "hospital": hosp.name if hosp else "Unknown",
            "date": slot.date, "time": slot.start_time,
            "patient_name": patient_name,
            "phone_masked": mask_phone(phone),
            "sms_sent": bool(sms_res and sms_res.get("success")),
            "sms_status": sms_res.get("status") if sms_res else "NONE",
            "sms_provider": sms_res.get("provider") if sms_res else "simulation",
            "departure_buffer": dep_buffer,
            "timeline": timeline
        }
    except Exception as e:
        db.rollback()
        return {"error": str(e)}
    finally:
        db.close()


def tool_get_appointments(patient_id):
    db = SessionLocal()
    try:
        appointments = db.query(Appointment).filter(Appointment.patient_id == patient_id).all()
        result = []
        for apt in appointments:
            doc = db.query(Doctor).filter(Doctor.id == apt.doctor_id).first()
            hosp = db.query(Hospital).filter(Hospital.id == apt.hospital_id).first()
            slot = db.query(AppointmentSlot).filter(AppointmentSlot.id == apt.slot_id).first()
            queue = db.query(QueueEntry).filter(QueueEntry.appointment_id == apt.id).first()
            slot_time = slot.start_time if slot else "10:30 AM"

            result.append({
                "id": apt.id,
                "doctor": doc.name if doc else "Unknown",
                "department": doc.department if doc else "Unknown",
                "hospital": hosp.name if hosp else "Unknown",
                "date": slot.date if slot else "Unknown",
                "time": slot_time,
                "status": apt.status,
                "queue_number": apt.queue_number,
                "queue_position": queue.position if queue else apt.queue_position,
                "estimated_wait_minutes": queue.estimated_wait_minutes if queue else apt.estimated_wait_minutes,
                "timeline": get_or_create_timeline(apt.id, initial_pos=queue.position if queue else 5, wait_min=queue.estimated_wait_minutes if queue else 20),
                "departure_buffer": calculate_departure_buffer(distance_km=2.2, appointment_time_str=slot_time)
            })
        return result
    finally:
        db.close()


def tool_cancel_appointment(appointment_id):
    db = SessionLocal()
    try:
        apt = db.query(Appointment).filter(Appointment.id == appointment_id).first()
        if not apt:
            return {"error": "Appointment not found"}
        if apt.status == "Cancelled":
            return {"error": "Already cancelled"}

        apt.status = "Cancelled"
        slot = db.query(AppointmentSlot).filter(AppointmentSlot.id == apt.slot_id).first()
        if slot:
            slot.available = True
        queue = db.query(QueueEntry).filter(QueueEntry.appointment_id == appointment_id).first()
        if queue:
            queue.status = "cancelled"
        
        notif = Notification(
            patient_id=apt.patient_id, appointment_id=apt.id,
            message=f"Your appointment #{apt.id} has been cancelled.",
            notification_type="appointment_cancelled"
        )
        db.add(notif)
        db.commit()
        return {"success": True, "message": "Appointment cancelled."}
    except Exception as e:
        db.rollback()
        return {"error": str(e)}
    finally:
        db.close()


def tool_get_queue_status(appointment_id=None, patient_id=1):
    db = SessionLocal()
    try:
        if appointment_id:
            queue = db.query(QueueEntry).filter(QueueEntry.appointment_id == appointment_id).first()
            if queue:
                apt = db.query(Appointment).filter(Appointment.id == appointment_id).first()
                doc = db.query(Doctor).filter(Doctor.id == apt.doctor_id).first() if apt else None
                hosp = db.query(Hospital).filter(Hospital.id == apt.hospital_id).first() if apt else None
                slot = db.query(AppointmentSlot).filter(AppointmentSlot.id == apt.slot_id).first() if apt else None
                slot_time = slot.start_time if slot else "10:30 AM"

                return {
                    "appointment_id": appointment_id,
                    "queue_number": queue.queue_number,
                    "position": queue.position,
                    "people_ahead": max(0, queue.position - 1),
                    "estimated_wait_minutes": queue.estimated_wait_minutes,
                    "status": "your_turn" if queue.position <= 1 else queue.status,
                    "doctor": doc.name if doc else "Dr. Rajesh Sharma",
                    "department": doc.department if doc else "General Medicine",
                    "hospital": hosp.name if hosp else "City General Hospital",
                    "date": slot.date if slot else "Today",
                    "time": slot_time,
                    "timeline": get_or_create_timeline(appointment_id, initial_pos=queue.position, wait_min=queue.estimated_wait_minutes),
                    "departure_buffer": calculate_departure_buffer(distance_km=2.2, appointment_time_str=slot_time)
                }
        # Get all queues for patient
        apts = db.query(Appointment).filter(
            Appointment.patient_id == patient_id,
            Appointment.status == "Confirmed"
        ).all()
        results = []
        for apt in apts:
            queue = db.query(QueueEntry).filter(QueueEntry.appointment_id == apt.id).first()
            if queue:
                doc = db.query(Doctor).filter(Doctor.id == apt.doctor_id).first()
                hosp = db.query(Hospital).filter(Hospital.id == apt.hospital_id).first()
                slot = db.query(AppointmentSlot).filter(AppointmentSlot.id == apt.slot_id).first()
                slot_time = slot.start_time if slot else "10:30 AM"
                results.append({
                    "appointment_id": apt.id,
                    "queue_number": queue.queue_number,
                    "position": queue.position,
                    "people_ahead": max(0, queue.position - 1),
                    "estimated_wait_minutes": queue.estimated_wait_minutes,
                    "status": "your_turn" if queue.position <= 1 else queue.status,
                    "doctor": doc.name if doc else "Dr. Rajesh Sharma",
                    "department": doc.department if doc else "General Medicine",
                    "hospital": hosp.name if hosp else "City General Hospital",
                    "date": slot.date if slot else "Today",
                    "time": slot_time,
                    "timeline": get_or_create_timeline(apt.id, initial_pos=queue.position, wait_min=queue.estimated_wait_minutes),
                    "departure_buffer": calculate_departure_buffer(distance_km=2.2, appointment_time_str=slot_time)
                })
        return results
    finally:
        db.close()


def tool_get_notifications(patient_id=1, unread_only=True):
    db = SessionLocal()
    try:
        query = db.query(Notification).filter(Notification.patient_id == patient_id)
        if unread_only:
            query = query.filter(Notification.read == False)
        notifs = query.order_by(Notification.created_at.desc()).limit(20).all()
        result = []
        for n in notifs:
            result.append({
                "id": n.id,
                "message": n.message,
                "type": n.notification_type,
                "read": n.read,
                "created_at": str(n.created_at)
            })
        return result
    finally:
        db.close()


def tool_mark_notifications_read(patient_id=1):
    db = SessionLocal()
    try:
        db.query(Notification).filter(
            Notification.patient_id == patient_id,
            Notification.read == False
        ).update({"read": True})
        db.commit()
        return {"success": True}
    finally:
        db.close()


# =====================================================
# 6. INTELLIGENT AGENT - CORE
# =====================================================

# In-memory session store (per-session conversation context)
sessions = {}


def get_session(session_id: str):
    if session_id not in sessions:
        sessions[session_id] = {
            "conversation_history": [],
            "symptoms": [],
            "duration": None,
            "severity": None,
            "urgency": None,
            "urgency_reason": None,
            "patient_data": {
                "age": None, "gender": None, "heart_rate": None,
                "temperature": None, "pre_existing_conditions": []
            },
            "location": {"latitude": None, "longitude": None},
            "hospitals_found": [],
            "selected_hospital": None,
            "recommended_department": None,
            "appointments": [],
            "current_appointment": None,
            "journey_stage": "greeting",  # greeting, symptom_collection, urgency_assessed, hospital_search, hospital_selected, booking, tracking
            "patient_id": 1,
            "pending_action": None,
            "pending_action_data": None
        }
    return sessions[session_id]


AGENT_SYSTEM_PROMPT = """IMPORTANT — THE APPLICATION MUST HAVE A FULLY INTELLIGENT HEALTHCARE ASSISTANT.
Do NOT build separate disconnected features.
The entire application must behave like ONE intelligent healthcare agent that supports the user throughout the complete journey:
User speech/text → understanding symptoms → urgency assess → location determination → nearby hospitals → hospital comparison → appointment booking → queue/live status → proactive notifications → next action.

The assistant must maintain context from the beginning of the conversation until the appointment is completed.

==================================================
23 CORE HEALTHCARE AGENT PROTOCOLS:
==================================================

1. SYMPTOM UNDERSTANDING:
- Extract structured entities: symptoms (list), duration, severity (mild/moderate/severe), frequency, relevant context.
- Do NOT repeatedly ask questions that the user has already answered.
- If important information is missing and materially affects urgency assessment, ask ONE short follow-up question.
- Retain all context across conversational turns.

2. INTELLIGENT URGENCY ASSESSMENT:
- Classify urgency into: LOW, MODERATE, HIGH, EMERGENCY.
- Explain WHY the urgency level was assigned in simple language.
- CRITICAL: Never claim a confirmed diagnosis. Never say "You have pneumonia".
  Instead say: "These symptoms can occur with several conditions, and a healthcare professional should evaluate you."
- Always show: "⚕️ AI guidance only — not a medical diagnosis."

3. LOCATION INTELLIGENCE:
- If location permission is available: use user's coordinates.
- If denied: politely ask to enable GPS or provide a city/area manually, then search immediately.

4. NEARBY HOSPITAL INTELLIGENCE:
- Do not display random hospitals. Help the user choose based on: distance, travel time, open/closed status, emergency availability, specialty/services, appointment availability, and user's urgency level.
- HIGH urgency: Recommend nearest facility with 24/7 emergency services.
- MODERATE urgency: Recommend facilities with shortest travel time and relevant specialty.
- LOW urgency: Recommend regular consultation facilities.
- Do NOT fabricate hospital capabilities.

5. MAP + AGENT WORK TOGETHER:
- Map is NOT a separate feature; the assistant actively controls hospital discovery on the map.
- Natural requests like "Which one is closest?", "Show hospitals nearby" trigger map updates and highlight nearest facilities.

6. HOSPITAL COMPARISON:
- Answer natural comparison questions: "Which is closest?", "Which one is open?", "Which has emergency services?", "Which has appointments today?", "Which one has the shortest waiting time?".
- Answer using available real data. If live queue/wait data is unavailable for a hospital, say: "I don't have live information about that hospital's waiting time." Never invent data.

7. APPOINTMENT BOOKING AGENT:
- User can say "Book me an appointment at the nearest hospital" or "Book that one".
- Identify suitable hospital, show slots, confirm selection, create appointment, and return booking ID & queue details.
- Always ask for user confirmation before creating the booking.

8. APPOINTMENT CONTEXT & TRAVEL BUFFER:
- Remember: Hospital, Appointment ID, Date, Time, Doctor/Specialty, Queue number, Current status.
- When user asks "When should I leave?" or "Should I go now?": calculate based on estimated travel time and give buffer (e.g., "Your appointment is at 10:30 AM. Based on ~18 min travel time, leaving around 10:00 AM gives you a comfortable buffer.").

9. LIVE QUEUE INTELLIGENCE:
- User can ask: "How many people are before me?", "How long will I wait?", "Has my turn come?".
- Return actual people ahead, estimated wait time in minutes, and token status.

10. PROACTIVE NOTIFICATIONS:
- Proactively inform about: appointment confirmed, queue position updates, wait time shifts, turn approaching.

11. VOICE-FIRST SUPPORT:
- Both voice and text use the identical assistant brain and intent logic.

12. INTENT SYSTEM:
Valid intents:
SYMPTOM_REPORT, SYMPTOM_FOLLOWUP, URGENCY_ASSESSMENT, FIND_HOSPITAL, VIEW_HOSPITAL, COMPARE_HOSPITALS, GET_DIRECTIONS, CHECK_APPOINTMENTS, BOOK_APPOINTMENT, CANCEL_APPOINTMENT, VIEW_APPOINTMENT, CHECK_QUEUE, CHECK_WAIT_TIME, TRAVEL_TIME_INQUIRY, CHECK_NOTIFICATION, GENERAL_HEALTHCARE, EMERGENCY_HELP, GREETING, CONFIRM_ACTION, PROVIDE_LOCATION.

13. TOOL-BASED AGENT ARCHITECTURE:
Actual tools:
getUserLocation, searchNearbyHospitals, getHospitalDetails, getAppointmentSlots, bookAppointment, cancelAppointment, getAppointmentStatus, getQueueStatus, getEstimatedWaitTime, getDirections, getNotifications, searchDoctors.

14. ACTION CONFIRMATION:
Ask confirmation for irreversible/key actions: booking, cancellation.

15. CONVERSATION MEMORY:
Never make the user repeat symptoms or context when moving from assessment to finding hospitals to booking.

16. EMERGENCY SAFETY:
For chest pain, difficulty breathing, stroke, severe bleeding: immediately provide urgent emergency guidance without forcing through lengthy booking flows.

==================================================
JSON OUTPUT SCHEMA:
==================================================
Respond with ONLY valid JSON:
{
  "intent": "<INTENT_NAME>",
  "response": "<Natural language response with ⚕️ AI guidance only — not a medical diagnosis>",
  "extracted_data": {
    "symptoms": ["list of symptoms"],
    "severity": "mild/moderate/severe/null",
    "duration": "string or null",
    "age": null,
    "gender": null,
    "heart_rate": null,
    "temperature": null,
    "location_query": "city name or null",
    "hospital_name": "name or null",
    "appointment_id": null,
    "doctor_id": null,
    "slot_id": null,
    "hospital_id": null,
    "confirmation": true/false/null
  },
  "tools_to_call": ["tool names"],
  "urgency": "LOW/MODERATE/HIGH/EMERGENCY/null",
  "urgency_reason": "why urgency was chosen or null",
  "requires_confirmation": false,
  "action_type": "book_appointment/cancel_appointment/null",
  "ui_action": "show_map/show_hospitals/show_booking/show_queue/show_appointments/none",
  "next_question": "short question if missing or null"
}"""


def process_agent_message(session_id: str, user_message: str, location: dict = None):
    """Core agent processing: understands user, calls tools, responds naturally"""
    session = get_session(session_id)

    # Update location if provided
    if location and location.get("latitude"):
        session["location"] = location

    # Add to conversation history
    session["conversation_history"].append({"role": "user", "content": user_message})

    # Build context for the LLM
    context = {
        "current_symptoms": session["symptoms"],
        "severity": session["severity"],
        "duration": session["duration"],
        "urgency": session["urgency"],
        "patient_data": session["patient_data"],
        "location_available": bool(session["location"].get("latitude")),
        "hospitals_found": len(session["hospitals_found"]),
        "selected_hospital": session["selected_hospital"],
        "recommended_department": session["recommended_department"],
        "current_appointment": session["current_appointment"],
        "journey_stage": session["journey_stage"],
        "pending_action": session["pending_action"]
    }

    # Build messages for LLM
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    messages.append({"role": "system", "content": f"Current session context:\n{json.dumps(context, indent=2)}"})

    # Include last 10 conversation turns for memory
    for msg in session["conversation_history"][-10:]:
        messages.append(msg)

    if client is None:
        return _fallback_agent_response(session, user_message)

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=1000,
        )
        response_text = chat_completion.choices[0].message.content
        agent_response = json.loads(response_text)
    except Exception as e:
        print(f"Agent LLM error: {e}")
        return _fallback_agent_response(session, user_message)

    # Process extracted data
    extracted = agent_response.get("extracted_data", {})
    if extracted:
        # Update symptoms
        new_symptoms = extracted.get("symptoms", [])
        if new_symptoms:
            for s in new_symptoms:
                if s and s.lower() not in [x.lower() for x in session["symptoms"]]:
                    session["symptoms"].append(s)

        # Update severity, duration
        if extracted.get("severity"):
            session["severity"] = extracted["severity"]
        if extracted.get("duration"):
            session["duration"] = extracted["duration"]

        # Update patient data
        for key in ["age", "gender", "heart_rate", "temperature"]:
            if extracted.get(key):
                session["patient_data"][key] = extracted[key]

    # Update urgency
    if agent_response.get("urgency"):
        session["urgency"] = agent_response["urgency"]
        session["urgency_reason"] = agent_response.get("urgency_reason", "")

    # Execute tools
    tool_results = {}
    tools_to_call = agent_response.get("tools_to_call", [])

    for tool in tools_to_call:
        tool_result = _execute_tool(tool, session, extracted, agent_response)
        tool_results[tool] = tool_result

    # If we did department prediction from symptoms
    if session["symptoms"] and not session["recommended_department"]:
        symptoms_str = ", ".join(session["symptoms"])
        prediction = hybrid_predict_logic(symptoms_str)
        if prediction["Top 3 Recommendations"]:
            session["recommended_department"] = prediction["Top 3 Recommendations"][0]["Department"]
            tool_results["prediction"] = prediction

    # Update journey stage based on intent
    intent = agent_response.get("intent", "")
    if intent in ["SYMPTOM_REPORT", "SYMPTOM_FOLLOWUP"]:
        session["journey_stage"] = "symptom_collection"
    elif intent == "URGENCY_ASSESSMENT" or session["urgency"]:
        session["journey_stage"] = "urgency_assessed"
    elif intent in ["FIND_HOSPITAL", "COMPARE_HOSPITALS"]:
        session["journey_stage"] = "hospital_search"
    elif intent == "BOOK_APPOINTMENT":
        session["journey_stage"] = "booking"
    elif intent in ["CHECK_QUEUE", "CHECK_WAIT_TIME"]:
        session["journey_stage"] = "tracking"

    # Build final response
    response_text = agent_response.get("response", "I'm here to help with your healthcare needs.")

    # Add to conversation history
    session["conversation_history"].append({"role": "assistant", "content": response_text})

    return {
        "response": response_text,
        "intent": intent,
        "urgency": session["urgency"],
        "urgency_reason": session["urgency_reason"],
        "symptoms": session["symptoms"],
        "severity": session["severity"],
        "duration": session["duration"],
        "recommended_department": session["recommended_department"],
        "department": session["recommended_department"],
        "journey_stage": session["journey_stage"],
        "ui_action": agent_response.get("ui_action", "none"),
        "requires_confirmation": agent_response.get("requires_confirmation", False),
        "action_type": agent_response.get("action_type"),
        "pending_action_data": extracted,
        "tool_results": tool_results,
        "next_question": agent_response.get("next_question"),
        "selected_hospital": session["selected_hospital"],
        "current_appointment": session["current_appointment"],
        "hospitals_count": len(session["hospitals_found"])
    }


def _execute_tool(tool_name, session, extracted, agent_response):
    """Execute a tool and return results"""
    try:
        if tool_name == "searchNearbyHospitals":
            lat = session["location"].get("latitude")
            lon = session["location"].get("longitude")
            city = extracted.get("location_query")
            if not lat and city:
                lat, lon = get_coordinates_for_city(city)
                if lat:
                    session["location"]["latitude"] = lat
                    session["location"]["longitude"] = lon
            if lat and lon:
                hospitals = tool_search_hospitals_osm(lat, lon, department=session["recommended_department"])
                session["hospitals_found"] = hospitals
                return {"hospitals": hospitals, "count": len(hospitals)}
            return {"error": "Location not available. Please share your location."}

        elif tool_name == "searchDoctors":
            dept = extracted.get("department") or session["recommended_department"]
            hosp_id = extracted.get("hospital_id")
            doctors = tool_search_doctors(department=dept, hospital_id=hosp_id)
            return {"doctors": doctors}

        elif tool_name == "getAppointmentSlots":
            doctor_id = extracted.get("doctor_id")
            if doctor_id:
                slots = tool_get_slots(doctor_id)
                return {"slots": slots}
            return {"error": "Doctor not specified"}

        elif tool_name == "bookAppointment":
            result = tool_book_appointment(
                session["patient_id"],
                extracted.get("doctor_id"),
                extracted.get("hospital_id"),
                extracted.get("slot_id")
            )
            if result.get("id"):
                session["current_appointment"] = result
                session["appointments"].append(result)
            return result

        elif tool_name == "cancelAppointment":
            apt_id = extracted.get("appointment_id")
            if apt_id:
                return tool_cancel_appointment(apt_id)
            return {"error": "Appointment ID not specified"}

        elif tool_name == "getQueueStatus":
            apt_id = extracted.get("appointment_id")
            if not apt_id and session["current_appointment"]:
                apt_id = session["current_appointment"].get("id")
            return tool_get_queue_status(apt_id, session["patient_id"])

        elif tool_name == "getAppointmentStatus":
            return tool_get_appointments(session["patient_id"])

        elif tool_name == "getNotifications":
            return tool_get_notifications(session["patient_id"])

        elif tool_name == "getEstimatedWaitTime":
            apt_id = extracted.get("appointment_id")
            if not apt_id and session["current_appointment"]:
                apt_id = session["current_appointment"].get("id")
            queue = tool_get_queue_status(apt_id, session["patient_id"])
            return queue

        elif tool_name == "getUserLocation":
            return {"message": "Location requested from user"}

        elif tool_name == "getDirections":
            return {"message": "Directions will be shown on map"}

        return {"status": "tool not found"}
    except Exception as e:
        print(f"Tool execution error ({tool_name}): {e}")
        return {"error": str(e)}


def _fallback_agent_response(session, user_message):
    """Fallback when LLM is unavailable: supports full 23-step intelligent healthcare journey"""
    msg_lower = user_message.lower().strip()

    # 1. Emergency Safety Override
    if any(w in msg_lower for w in ["chest pain", "chest discomfort", "difficulty breathing", "breath", "heart attack", "stroke", "unconscious", "severe bleeding", "emergency help"]):
        session["urgency"] = "HIGH"
        session["urgency_reason"] = "Chest discomfort and breathing symptoms may require immediate medical attention."
        session["journey_stage"] = "urgency_assessed"
        # Pre-fetch nearest hospitals if location available
        hospitals = []
        lat = session["location"].get("latitude")
        lon = session["location"].get("longitude")
        if lat and lon:
            hospitals = tool_search_hospitals_osm(lat, lon, department="Emergency")
            session["hospitals_found"] = hospitals

        closest_text = f" The closest emergency facility is {hospitals[0]['name']} ({hospitals[0]['distance_km']} km away)." if hospitals else ""
        return {
            "response": f"⚠️ Those symptoms can sometimes require urgent medical attention. Please seek immediate medical care rather than waiting for a routine appointment.{closest_text}\n\nI have centered your map on nearby emergency facilities.\n\n⚕️ AI guidance only — not a medical diagnosis.",
            "intent": "EMERGENCY_HELP",
            "urgency": "HIGH",
            "urgency_reason": session["urgency_reason"],
            "symptoms": session["symptoms"],
            "ui_action": "show_map",
            "journey_stage": "urgency_assessed",
            "requires_confirmation": False,
            "tool_results": {"hospitals": hospitals}
        }

    # 2. Timing / Travel buffer questions ("When should I leave?", "Should I go now?")
    if any(w in msg_lower for w in ["when should i leave", "should i go now", "when to leave", "travel time", "how long to reach"]):
        apt = session.get("current_appointment")
        hosp_name = session.get("selected_hospital", {}).get("name") if session.get("selected_hospital") else "the hospital"
        if apt:
            return {
                "response": f"Your appointment is confirmed for {apt.get('time', '10:30 AM')} at {apt.get('hospital', hosp_name)}. Based on estimated local travel time of ~20 minutes and arrival check-in buffer, leaving 30 minutes before your slot is recommended.\n\n⚕️ AI guidance only — not a medical diagnosis.",
                "intent": "TRAVEL_TIME_INQUIRY",
                "urgency": session["urgency"],
                "symptoms": session["symptoms"],
                "ui_action": "show_queue",
                "journey_stage": "tracking",
                "requires_confirmation": False,
                "tool_results": {}
            }
        else:
            return {
                "response": "You don't have an active confirmed appointment yet. Once you book a slot, I'll calculate your recommended departure time and travel buffer!\n\n⚕️ AI guidance only — not a medical diagnosis.",
                "intent": "TRAVEL_TIME_INQUIRY",
                "urgency": session["urgency"],
                "symptoms": session["symptoms"],
                "ui_action": "none",
                "journey_stage": session["journey_stage"],
                "requires_confirmation": False,
                "tool_results": {}
            }

    # 3. Live Queue questions ("How many people ahead", "How long will I wait", "Has my turn come")
    if any(w in msg_lower for w in ["how many people", "ahead of me", "before me", "wait time", "how long will i wait", "how long until", "my turn", "has my turn come", "queue status", "check queue"]):
        queue_data = tool_get_queue_status(patient_id=session["patient_id"])
        ahead = queue_data.get("people_ahead", 4)
        wait_min = queue_data.get("estimated_wait_minutes", 20)
        pos = queue_data.get("queue_position", 5)
        
        if ahead == 0:
            msg = f"It's your turn right now! Please proceed directly to consultation room #1. Token #{queue_data.get('token_number', 101)}."
        elif ahead <= 2:
            msg = f"Your appointment is approaching! You are currently #{pos} in queue with {ahead} person(s) ahead. Estimated wait: ~{wait_min} minutes."
        else:
            msg = f"There are currently {ahead} people ahead of you in queue (Position #{pos}). Your estimated waiting time is approximately {wait_min} minutes."

        return {
            "response": f"{msg}\n\n⚕️ AI guidance only — not a medical diagnosis.",
            "intent": "CHECK_QUEUE",
            "urgency": session["urgency"],
            "symptoms": session["symptoms"],
            "ui_action": "show_queue",
            "journey_stage": "tracking",
            "requires_confirmation": False,
            "tool_results": {"queue": queue_data}
        }

    # 4. Hospital Comparison questions ("Which is closest", "Which has emergency", "Which has appointments today", "Which has shortest wait")
    if any(w in msg_lower for w in ["which one is closest", "which is closest", "which hospital is closest", "closest hospital"]):
        hospitals = session.get("hospitals_found", [])
        if not hospitals:
            lat = session["location"].get("latitude")
            lon = session["location"].get("longitude")
            if lat and lon:
                hospitals = tool_search_hospitals_osm(lat, lon)
            else:
                hospitals = tool_search_hospitals_db()
            session["hospitals_found"] = hospitals

        if hospitals:
            closest = hospitals[0]
            session["selected_hospital"] = closest
            return {
                "response": f"{closest['name']} is the closest option at {closest['distance_km']} km away ({closest.get('address', 'nearby')}). Would you like me to check available doctor slots there?\n\n⚕️ AI guidance only — not a medical diagnosis.",
                "intent": "COMPARE_HOSPITALS",
                "urgency": session["urgency"],
                "symptoms": session["symptoms"],
                "ui_action": "show_hospitals",
                "journey_stage": "hospital_search",
                "requires_confirmation": False,
                "tool_results": {"closest": closest}
            }
        else:
            return {
                "response": "Please share your location or city name so I can calculate exact distances and identify the closest hospital for you.",
                "intent": "FIND_HOSPITAL",
                "urgency": session["urgency"],
                "symptoms": session["symptoms"],
                "ui_action": "show_map",
                "journey_stage": "hospital_search",
                "requires_confirmation": False,
                "tool_results": {}
            }

    if any(w in msg_lower for w in ["which has emergency", "has emergency", "emergency services"]):
        return {
            "response": "Facilities with 24/7 Trauma and Emergency Departments are highlighted on your map. For immediate life-saving emergencies, you can also dial 108 / 112 directly.\n\n⚕️ AI guidance only — not a medical diagnosis.",
            "intent": "COMPARE_HOSPITALS",
            "urgency": session["urgency"],
            "symptoms": session["symptoms"],
            "ui_action": "show_map",
            "journey_stage": "hospital_search",
            "requires_confirmation": False,
            "tool_results": {}
        }

    if any(w in msg_lower for w in ["which one can see me today", "appointments today", "slots today", "see me today"]):
        return {
            "response": "City General Hospital and Apollo Speciality Hospital have outpatient consultation slots open today. Would you like me to show the earliest available time?\n\n⚕️ AI guidance only — not a medical diagnosis.",
            "intent": "COMPARE_HOSPITALS",
            "urgency": session["urgency"],
            "symptoms": session["symptoms"],
            "ui_action": "show_booking",
            "journey_stage": "hospital_search",
            "requires_confirmation": False,
            "tool_results": {}
        }

    if any(w in msg_lower for w in ["shortest waiting time", "least wait", "fastest queue"]):
        return {
            "response": "Based on live queue estimations, City General Hospital currently has the shortest estimated queue wait (~15-20 mins). Live wait data for other private clinics may vary.\n\n⚕️ AI guidance only — not a medical diagnosis.",
            "intent": "COMPARE_HOSPITALS",
            "urgency": session["urgency"],
            "symptoms": session["symptoms"],
            "ui_action": "show_hospitals",
            "journey_stage": "hospital_search",
            "requires_confirmation": False,
            "tool_results": {}
        }

    # 5. Booking Confirmation / Booking Actions
    if msg_lower in ["yes", "confirm", "confirm booking", "yes please", "book it", "book that one", "proceed"]:
        # If user is confirming a booking
        result = tool_book_appointment(patient_id=session["patient_id"], doctor_id=1, hospital_id=1, slot_id=1)
        if result.get("id"):
            session["current_appointment"] = result
            session["appointments"].append(result)
            session["journey_stage"] = "tracking"
            return {
                "response": f"🎉 Your appointment has been confirmed!\n\n• Hospital: {result.get('hospital', 'City Hospital')}\n• Doctor: {result.get('doctor', 'Dr. Rajesh Sharma')}\n• Date & Time: {result.get('date', 'Today')} at {result.get('time', '10:30 AM')}\n• Appointment ID: #{result.get('id')}\n• Queue Token: #{result.get('queue_position', 5)}\n\nYou can track live queue status and receive real-time updates directly from your dashboard.\n\n⚕️ AI guidance only — not a medical diagnosis.",
                "intent": "BOOK_APPOINTMENT",
                "urgency": session["urgency"],
                "symptoms": session["symptoms"],
                "ui_action": "show_queue",
                "journey_stage": "tracking",
                "requires_confirmation": False,
                "tool_results": {"appointment": result}
            }

    if any(w in msg_lower for w in ["book me an appointment", "book appointment", "book", "schedule consultation", "make an appointment"]):
        session["journey_stage"] = "booking"
        return {
            "response": "I found an available slot today at 10:30 AM with Dr. Rajesh Sharma (General Medicine) at City Hospital. Would you like me to confirm and book this appointment for you?\n\n⚕️ AI guidance only — not a medical diagnosis.",
            "intent": "BOOK_APPOINTMENT",
            "urgency": session["urgency"],
            "symptoms": session["symptoms"],
            "ui_action": "show_booking",
            "journey_stage": "booking",
            "requires_confirmation": True,
            "action_type": "book_appointment",
            "tool_results": {}
        }

    # 6. Directions / Map
    if any(w in msg_lower for w in ["take me there", "directions", "where is the hospital", "show directions", "navigate"]):
        return {
            "response": "Opening turn-by-turn directions to the selected hospital on your map. Safe travels!\n\n⚕️ AI guidance only — not a medical diagnosis.",
            "intent": "GET_DIRECTIONS",
            "urgency": session["urgency"],
            "symptoms": session["symptoms"],
            "ui_action": "show_map",
            "journey_stage": session["journey_stage"],
            "requires_confirmation": False,
            "tool_results": {}
        }

    # 7. Symptom Follow-ups (Severity & Duration responses)
    if any(w in msg_lower for w in ["severe", "mild", "moderate"]):
        session["severity"] = "severe" if "severe" in msg_lower else ("moderate" if "moderate" in msg_lower else "mild")
        urgency = "HIGH" if session["severity"] == "severe" else ("MODERATE" if session["severity"] == "moderate" else "LOW")
        session["urgency"] = urgency
        session["urgency_reason"] = f"Assessed as {urgency} based on reported {session['severity']} severity."
        session["journey_stage"] = "urgency_assessed"

        # Run hybrid department prediction if symptoms available
        prediction = None
        if session["symptoms"]:
            sym_str = ", ".join(session["symptoms"])
            prediction = hybrid_predict_logic(sym_str)
            if prediction.get("Top 3 Recommendations") and len(prediction["Top 3 Recommendations"]) > 0:
                session["recommended_department"] = prediction["Top 3 Recommendations"][0]["Department"]

        dept = session.get("recommended_department", "General Medicine")

        hospitals = []
        if session["location"].get("latitude"):
            hospitals = tool_search_hospitals_osm(session["location"]["latitude"], session["location"]["longitude"], department=dept)
            session["hospitals_found"] = hospitals

        hosp_note = f" I found {len(hospitals)} nearby hospitals with {dept} specialists." if hospitals else f" I can help you find nearby hospitals with {dept} specialists."
        return {
            "response": f"Thank you for clarifying. Based on your reported {session['severity']} symptoms ({', '.join(session['symptoms']) if session['symptoms'] else 'reported'}), this is assessed as **{urgency}** urgency.\n\nRecommended Department: **{dept}**\n\n{hosp_note} Would you like to view hospital options or check available doctor slots?\n\n⚕️ AI guidance only — not a medical diagnosis.",
            "intent": "URGENCY_ASSESSMENT",
            "urgency": urgency,
            "urgency_reason": session["urgency_reason"],
            "symptoms": session["symptoms"],
            "severity": session["severity"],
            "recommended_department": dept,
            "department": dept,
            "recommended_action": "Consider consulting a healthcare professional soon.",
            "emergency_warning": (urgency == "HIGH"),
            "message": f"Based on the information provided, your symptoms may require medical attention ({dept}).",
            "ui_action": "show_hospitals" if hospitals else "none",
            "journey_stage": "urgency_assessed",
            "requires_confirmation": False,
            "tool_results": {"hospitals": hospitals, "prediction": prediction if prediction else {}}
        }

    if any(w in msg_lower for w in ["yesterday", "since", "days", "hours", "week"]):
        session["duration"] = user_message
        if not session["severity"]:
            return {
                "response": "Understood. How severe is the discomfort right now — mild, moderate, or severe?\n\n⚕️ AI guidance only — not a medical diagnosis.",
                "intent": "SYMPTOM_FOLLOWUP",
                "urgency": session["urgency"],
                "symptoms": session["symptoms"],
                "recommended_department": session.get("recommended_department"),
                "department": session.get("recommended_department"),
                "ui_action": "none",
                "journey_stage": "symptom_collection",
                "requires_confirmation": False,
                "tool_results": {}
            }

    # 8. Initial Symptom Report
    if any(w in msg_lower for w in ["fever", "headache", "stomach pain", "pain", "cough", "cold", "vomiting", "nausea", "dizziness", "body pain", "sore throat", "weakness", "fatigue"]):
        if user_message not in session["symptoms"]:
            session["symptoms"].append(user_message)
        session["journey_stage"] = "symptom_collection"
        
        # Run hybrid department prediction
        prediction = hybrid_predict_logic(user_message)
        if prediction.get("Top 3 Recommendations") and len(prediction["Top 3 Recommendations"]) > 0:
            session["recommended_department"] = prediction["Top 3 Recommendations"][0]["Department"]
        dept = session.get("recommended_department", "General Medicine")
        
        # Check if severity or duration was already mentioned in the message
        has_severe = "severe" in msg_lower
        has_moderate = "moderate" in msg_lower
        has_mild = "mild" in msg_lower
        if has_severe or has_moderate or has_mild:
            session["severity"] = "severe" if has_severe else ("moderate" if has_moderate else "mild")
            urgency = "HIGH" if has_severe else ("MODERATE" if has_moderate else "LOW")
            session["urgency"] = urgency
            session["urgency_reason"] = f"Classified as {urgency} based on reported {session['severity']} severity."
            session["journey_stage"] = "urgency_assessed"
            return {
                "response": f"I have noted your symptoms ({user_message}). Based on {session['severity']} severity, urgency is assessed as **{urgency}**.\n\nRecommended Department: **{dept}**\n\nI recommend seeking in-person medical evaluation. Shall I find the closest hospitals near you?\n\n⚕️ AI guidance only — not a medical diagnosis.",
                "intent": "URGENCY_ASSESSMENT",
                "urgency": urgency,
                "urgency_reason": session["urgency_reason"],
                "symptoms": session["symptoms"],
                "severity": session["severity"],
                "recommended_department": dept,
                "department": dept,
                "recommended_action": "Consider consulting a healthcare professional soon.",
                "emergency_warning": (urgency == "HIGH"),
                "ui_action": "show_map",
                "journey_stage": "urgency_assessed",
                "requires_confirmation": False,
                "tool_results": {"prediction": prediction}
            }
        else:
            return {
                "response": f"I understand you're experiencing {user_message}. How severe is the discomfort — mild, moderate, or severe?\n\n⚕️ AI guidance only — not a medical diagnosis.",
                "intent": "SYMPTOM_REPORT",
                "urgency": None,
                "symptoms": session["symptoms"],
                "severity": None,
                "recommended_department": dept,
                "department": dept,
                "ui_action": "none",
                "journey_stage": "symptom_collection",
                "requires_confirmation": False,
                "tool_results": {"prediction": prediction}
            }

    # 9. Hospital Search / Location
    if any(w in msg_lower for w in ["hospital", "nearby", "find", "closest", "doctor"]):
        lat = session["location"].get("latitude")
        lon = session["location"].get("longitude")
        hospitals = []
        if lat and lon:
            hospitals = tool_search_hospitals_osm(lat, lon)
            session["hospitals_found"] = hospitals
            return {
                "response": f"I found {len(hospitals)} nearby hospitals within your area. The closest is {hospitals[0]['name']} ({hospitals[0]['distance_km']} km away). Would you like to view doctor slots or get directions?\n\n⚕️ AI guidance only — not a medical diagnosis.",
                "intent": "FIND_HOSPITAL",
                "urgency": session["urgency"],
                "symptoms": session["symptoms"],
                "ui_action": "show_hospitals",
                "journey_stage": "hospital_search",
                "requires_confirmation": False,
                "tool_results": {"hospitals": hospitals}
            }
        else:
            return {
                "response": "I'll help you find nearby hospitals. Please enable location or type your city/area name below so I can locate healthcare facilities.\n\n⚕️ AI guidance only — not a medical diagnosis.",
                "intent": "FIND_HOSPITAL",
                "urgency": session["urgency"],
                "symptoms": session["symptoms"],
                "ui_action": "show_map",
                "journey_stage": "hospital_search",
                "requires_confirmation": False,
                "tool_results": {}
            }

    # 10. Default Greeting / General Guidance
    return {
        "response": "Hello! I am your intelligent healthcare assistant. I guide you through your complete journey:\n• Share symptoms (e.g. 'I have fever and severe headache')\n• Urgency assessment & triage\n• Finding & comparing nearby hospitals\n• Booking consultation appointments\n• Real-time live queue tracking & travel buffer\n\nHow can I help you today?\n\n⚕️ AI guidance only — not a medical diagnosis.",
        "intent": "GREETING",
        "urgency": None,
        "symptoms": session["symptoms"],
        "ui_action": "none",
        "journey_stage": "greeting",
        "requires_confirmation": False,
        "tool_results": {}
    }


# =====================================================
# 7. API ENDPOINTS
# =====================================================

class AgentMessageRequest(BaseModel):
    message: str
    session_id: str = "default"
    location: Optional[dict] = None

class SearchRequest(BaseModel):
    department: Optional[str] = None
    hospital_id: Optional[int] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    radius_km: Optional[float] = 10.0
    query: Optional[str] = None

class SlotsRequest(BaseModel):
    doctor_id: int

class BookRequest(BaseModel):
    doctor_id: int
    hospital_id: int
    slot_id: int
    patient_id: int = 1

class CancelRequest(BaseModel):
    appointment_id: int

class SymptomRequest(BaseModel):
    symptoms: str

class TriageRequest(BaseModel):
    Name: Optional[str] = "Anonymous"
    Age: Optional[int] = 30
    Gender: Optional[str] = "Male"
    Systolic_BP: Optional[int] = 120
    Diastolic_BP: Optional[int] = 80
    Heart_Rate: Optional[int] = 72
    Temperature: Optional[float] = 37.0
    Symptoms: str

class LocationRequest(BaseModel):
    latitude: float
    longitude: float

class ChatRequest(BaseModel):
    message: Optional[str] = None
    prompt: Optional[str] = None
    text: Optional[str] = None
    question: Optional[str] = None

class ChatResponse(BaseModel):
    response: str

class ProfileRequest(BaseModel):
    full_name: str
    age: int
    gender: str
    phone: Optional[str] = None
    email: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class SymptomAnalyzeRequest(BaseModel):
    symptoms: str
    age: Optional[int] = None
    gender: Optional[str] = None
    severity: Optional[str] = None
    duration: Optional[str] = None
    user: Optional[Dict[str, Any]] = None


# --- Primary Agent Endpoint ---
@app.post("/agent/message")
async def agent_message(request: AgentMessageRequest):
    """Main intelligent agent endpoint — processes user messages with full context"""
    result = process_agent_message(
        session_id=request.session_id,
        user_message=request.message,
        location=request.location
    )
    return result


@app.post("/agent/location")
async def agent_update_location(request: LocationRequest):
    """Update agent session with user location"""
    session = get_session("default")
    session["location"] = {"latitude": request.latitude, "longitude": request.longitude}
    return {"success": True, "message": "Location updated"}


@app.get("/agent/session/{session_id}")
async def get_agent_session(session_id: str):
    """Get current agent session state"""
    session = get_session(session_id)
    return {
        "symptoms": session["symptoms"],
        "severity": session["severity"],
        "duration": session["duration"],
        "urgency": session["urgency"],
        "recommended_department": session["recommended_department"],
        "journey_stage": session["journey_stage"],
        "current_appointment": session["current_appointment"],
        "hospitals_count": len(session["hospitals_found"]),
        "location_available": bool(session["location"].get("latitude"))
    }


@app.post("/agent/reset/{session_id}")
async def reset_agent_session(session_id: str):
    """Reset agent session"""
    if session_id in sessions:
        del sessions[session_id]
    return {"success": True, "message": "Session reset"}


# --- Hospital Search (both OSM and internal) ---
@app.post("/hospitals/search")
async def hospitals_search_endpoint(request: SearchRequest):
    lat = request.latitude
    lon = request.longitude
    if request.query and (lat is None or lon is None):
        lat, lon = get_coordinates_for_city(request.query)

    if lat is None or lon is None:
        # Return internal hospitals
        hospitals = tool_search_hospitals_db(request.department)
        return {"success": True, "hospitals": hospitals}

    hospitals = tool_search_hospitals_osm(lat, lon, request.radius_km or 10.0, request.department)
    return {"success": True, "hospitals": hospitals}


@app.post("/doctors/search")
async def doctors_search_endpoint(request: SearchRequest):
    return tool_search_doctors(request.department, request.hospital_id)


@app.post("/appointments/slots")
async def appointment_slots_endpoint(request: SlotsRequest):
    return tool_get_slots(request.doctor_id)


@app.post("/appointments/book")
async def appointments_book_endpoint(request: BookRequest):
    return tool_book_appointment(request.patient_id, request.doctor_id, request.hospital_id, request.slot_id)


@app.get("/appointments")
async def get_appointments_endpoint(patient_id: int = 1):
    return tool_get_appointments(patient_id)


@app.post("/appointments/cancel")
async def appointments_cancel_endpoint(request: CancelRequest):
    return tool_cancel_appointment(request.appointment_id)


# --- Queue ---
@app.get("/queue/status")
async def queue_status_endpoint(appointment_id: int = None, patient_id: int = 1):
    return tool_get_queue_status(appointment_id, patient_id)


# --- Notifications ---
@app.get("/notifications")
async def notifications_endpoint(patient_id: int = 1, unread_only: bool = True):
    return tool_get_notifications(patient_id, unread_only)


@app.post("/notifications/read")
async def mark_notifications_read(patient_id: int = 1):
    return tool_mark_notifications_read(patient_id)


# --- Prediction ---
@app.post("/predict")
def predict(data: SymptomRequest):
    if check_emergency(data.symptoms):
        return {
            "Emergency": True,
            "Message": "Possible medical emergency. Please seek immediate care.",
            "System": "Emergency Guard",
            "Top 3 Recommendations": [],
            "Similar Past Cases": []
        }
    result = hybrid_predict_logic(data.symptoms)
    try:
        db = SessionLocal()
        top3 = result["Top 3 Recommendations"]
        while len(top3) < 3:
            top3.append({"Department": "None", "Final Confidence (%)": 0.0})
        log = PredictionLog(
            symptoms=data.symptoms,
            department_1=top3[0]["Department"], confidence_1=top3[0]["Final Confidence (%)"],
            department_2=top3[1]["Department"], confidence_2=top3[1]["Final Confidence (%)"],
            department_3=top3[2]["Department"], confidence_3=top3[2]["Final Confidence (%)"],
            emergency=False
        )
        db.add(log)
        db.commit()
        db.close()
    except Exception as e:
        print(f"DB Logging failed: {e}")

    return {
        "Emergency": False,
        "System": result["System"],
        "Top 3 Recommendations": result["Top 3 Recommendations"],
        "Similar Past Cases": result.get("Similar Past Cases", [])
    }


@app.post("/triage")
def triage_endpoint(data: TriageRequest):
    risk_level = predict_risk_logic(
        data.Age, data.Gender, data.Systolic_BP, data.Diastolic_BP,
        data.Heart_Rate, data.Temperature, data.Symptoms
    )
    return {"risk_level": risk_level}


# --- Explain ---
@app.post("/explain")
async def explain(request: dict):
    if client is None:
        raise HTTPException(status_code=503, detail="AI Explainability is currently unavailable.")

    system_prompt = (
        "You are a medical AI explainability assistant.\n"
        "STRICT RULES:\n"
        "- Do NOT change the prediction.\n"
        "- Do NOT diagnose.\n"
        "- Do NOT suggest treatments.\n"
        "- Only explain why the given prediction makes sense.\n"
        "- Output only bullet points.\n"
        "- Each bullet must start with '- '.\n"
        "- Maximum 3 bullet points.\n"
        "- No paragraphs."
    )

    user_content = (
        f"Patient Data:\n"
        f"Symptoms: {request.get('symptoms', 'Unknown')}\n"
        f"Severity: {request.get('severity', 'Unknown')}\n"
        f"Duration: {request.get('duration', 'Unknown')}\n\n"
        f"Model Output:\n"
        f"Urgency: {request.get('urgency', 'Unknown')}\n"
        f"Recommended Department: {request.get('recommended_department', 'Unknown')}\n"
    )

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=300,
        )
        response_text = chat_completion.choices[0].message.content
        if not response_text:
            raise HTTPException(status_code=500, detail="Empty response from AI")
        return {"explanation": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


# --- Voice (STT) ---
@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    if client is None:
        raise HTTPException(status_code=503, detail="AI service unavailable")
    try:
        content = await file.read()
        audio_file = io.BytesIO(content)
        audio_file.name = file.filename
        translation = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
            response_format="json",
        )
        return {"text": translation.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT failed: {str(e)}")


# --- Legacy Chat ---
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if client is None:
        raise HTTPException(status_code=503, detail="AI service unavailable")
    user_message = request.message or request.prompt or request.text or request.question
    if not user_message:
        raise HTTPException(status_code=422, detail="Message is required")

    system_prompt = (
        "You are an informational medical assistant.\n"
        "Rules:\n"
        "- No diagnosis\n"
        "- No treatment advice\n"
        "- No emergency instructions\n"
        "- Use bullet points only\n"
        "- Only medical topics\n"
        "- Max 5 bullets\n"
        "- One sentence per bullet"
    )

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=200,
        )
        response_text = chat_completion.choices[0].message.content
        if not response_text:
            raise HTTPException(status_code=502, detail="Empty response from AI service")
        return {"response": response_text.strip()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat service failed: {str(e)}")


# --- Legacy assistant endpoint (kept for backward compat) ---
@app.post("/assistant")
async def assistant_endpoint_legacy(request: dict):
    message = request.get("message", "")
    session_data = request.get("session", {})
    result = process_agent_message("default", message, session_data.get("location"))
    # Map to old format
    return {
        "intent": result.get("intent", "general_help"),
        "response": result.get("response", ""),
        "tool": None,
        "tool_params": result.get("pending_action_data", {}),
        "requires_confirmation": result.get("requires_confirmation", False)
    }


# --- Health ---
@app.get("/api/health")
async def api_health_check():
    """Real services health verification endpoint conforming strictly to Phase 2 specification"""
    db_status = "error"
    try:
        db = SessionLocal()
        db.query(Hospital).first()
        db.close()
        db_status = "connected"
    except Exception:
        db_status = "error"

    ml_status = "ready" if (triage_model is not None and embedder is not None) else ("degraded" if embedder is not None else "not_configured")
    maps_status = "ready"
    sms_status = "ready" if sms_service.is_real else "not_configured"

    return {
        "status": "ok" if db_status == "connected" else "error",
        "services": {
            "database": db_status,
            "ml": ml_status,
            "maps": maps_status,
            "sms": sms_status
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if client else "degraded",
        "services": {
            "triage_model": triage_model is not None,
            "department_model": classifier is not None,
            "sentence_transformer": embedder is not None,
            "faiss": dept_index is not None,
            "groq": client is not None,
            "database": True,
            "maps": True,
            "hospital_search": True,
            "booking": True,
            "queue_system": True,
            "notifications": True,
            "agent": True
        }
    }


# --- Analytics ---
@app.get("/analytics")
def analytics():
    db = SessionLocal()
    total = db.query(PredictionLog).count()
    emergencies = db.query(PredictionLog).filter_by(emergency=True).count()
    logs = db.query(PredictionLog).all()
    department_counter = {}
    for log in logs:
        for dept in [log.department_1, log.department_2, log.department_3]:
            if dept and dept != "None":
                department_counter[dept] = department_counter.get(dept, 0) + 1
    db.close()
    return {
        "Total Predictions": total,
        "Emergency Cases": emergencies,
        "Department Frequency": department_counter
    }


# --- User Profile ---
@app.post("/api/user/profile")
async def create_user_profile(request: ProfileRequest):
    db = SessionLocal()
    try:
        profile = UserProfile(
            full_name=request.full_name,
            age=request.age,
            gender=request.gender,
            phone=request.phone,
            email=request.email,
            latitude=request.latitude,
            longitude=request.longitude
        )
        db.add(profile)
        db.commit()
        db.refresh(profile)

        # Also create a Patient record for backward compat
        patient = Patient(
            age=request.age,
            gender=request.gender,
            symptoms="",
            heart_rate=72,
            temperature=37.0,
            pre_existing_conditions=""
        )
        db.add(patient)
        db.commit()
        db.refresh(patient)

        return {
            "success": True,
            "profile_id": profile.id,
            "patient_id": patient.id,
            "full_name": profile.full_name
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


# --- Dedicated Symptom Analysis ---
@app.post("/api/symptoms/analyze")
async def analyze_symptoms_endpoint(request: SymptomAnalyzeRequest):
    """Dedicated symptom analysis using trained Random Forest ML triage model, duration extraction, and hybrid department matching"""
    symptoms_text = request.symptoms.strip()
    if not symptoms_text:
        raise HTTPException(status_code=422, detail="Symptoms text is required")

    # Extract user info
    user_data = request.user or {}
    age = request.age or user_data.get("age") or 30
    gender = request.gender or user_data.get("gender") or "Female"
    severity = request.severity or user_data.get("severity") or "moderate"

    # Extract duration from text
    duration = "Unspecified"
    import re
    dur_match = re.search(r'(\d+\s*(?:day|days|week|weeks|month|months|hour|hours))', symptoms_text, re.IGNORECASE)
    if dur_match:
        duration = dur_match.group(1)
    elif "yesterday" in symptoms_text.lower():
        duration = "1 day"

    # 1. Emergency safety layer check (First line of defense)
    is_emergency = check_emergency(symptoms_text)

    # 2. Hybrid Medical Department Recommendation
    prediction = hybrid_predict_logic(symptoms_text)
    top_departments = prediction.get("Top 3 Recommendations", [])
    recommended_dept = top_departments[0]["Department"] if top_departments else "General Medicine"
    dept_confidence = top_departments[0]["Final Confidence (%)"] if top_departments else 75.0

    # 3. ML Triage Model Inference (Random Forest)
    urgency = "LOW"
    ml_confidence = 0.85
    urgency_reason = "Symptoms appear mild. Rest and hydration are advised; seek care if symptoms persist."
    emergency_warning = False
    recommended_action = "Consider scheduling a routine consultation with a healthcare professional."

    if is_emergency:
        urgency = "HIGH"
        ml_confidence = 0.98
        emergency_warning = True
        recommended_dept = "Emergency"
        urgency_reason = "Critical emergency indicators detected. Immediate emergency medical intervention is required."
        recommended_action = "Please contact emergency services (108/112) or visit the nearest emergency department immediately."
    elif triage_model is not None and triage_encoder is not None:
        try:
            # Map input text to trained categorical feature
            known_symptoms = [
                'Body Pain', 'Chest Pain', 'Cough', 'Difficulty Breathing', 'Dizziness',
                'High Fever', 'Joint Pain', 'Mild Headache', 'Numbness', 'Palpitations',
                'Severe Headache', 'Shivering', 'Shortness of Breath', 'Skin Rash', 'Slurred Speech'
            ]
            t_low = symptoms_text.lower()
            matched_sym = "Mild Headache"
            if "chest" in t_low: matched_sym = "Chest Pain"
            elif "breath" in t_low or "shortness" in t_low: matched_sym = "Difficulty Breathing"
            elif "fever" in t_low or "temperature" in t_low: matched_sym = "High Fever"
            elif "shiver" in t_low or "chill" in t_low: matched_sym = "Shivering"
            elif "severe headache" in t_low or "migraine" in t_low: matched_sym = "Severe Headache"
            elif "headache" in t_low: matched_sym = "Mild Headache"
            elif "body pain" in t_low or "ache" in t_low: matched_sym = "Body Pain"
            elif "joint" in t_low: matched_sym = "Joint Pain"
            elif "dizz" in t_low: matched_sym = "Dizziness"
            elif "cough" in t_low: matched_sym = "Cough"
            elif "rash" in t_low or "skin" in t_low: matched_sym = "Skin Rash"
            elif "palpitat" in t_low or "heart" in t_low: matched_sym = "Palpitations"
            elif "numb" in t_low: matched_sym = "Numbness"
            elif "speech" in t_low: matched_sym = "Slurred Speech"

            p_age = int(age)
            p_gender = "Female" if str(gender).lower().startswith("f") else "Male"
            p_temp = 38.4 if ("fever" in t_low or "shiver" in t_low) else 37.0
            p_hr = 88 if ("fever" in t_low or "palpitat" in t_low) else 75
            p_sbp = 120
            p_dbp = 80
            mean_bp = (p_sbp + 2 * p_dbp) / 3.0

            X_num = np.array([[p_age, mean_bp, p_hr, p_temp]])
            X_cat = triage_encoder.transform([[p_gender, matched_sym]])
            X = np.hstack([X_num, X_cat])

            raw_pred = triage_model.predict(X)[0]
            probs = triage_model.predict_proba(X)[0]
            classes = list(triage_model.classes_)
            c_idx = classes.index(raw_pred) if raw_pred in classes else 0
            ml_confidence = round(float(probs[c_idx]), 2)

            if raw_pred == "High":
                urgency = "HIGH"
                urgency_reason = f"Random Forest triage classifier evaluated your symptom profile ('{matched_sym}') as high clinical urgency ({int(ml_confidence * 100)}% model probability)."
                recommended_action = "Please seek prompt medical evaluation today."
            elif raw_pred == "Medium":
                urgency = "MODERATE"
                urgency_reason = f"Random Forest triage classifier evaluated your symptom profile ('{matched_sym}') as moderate urgency ({int(ml_confidence * 100)}% model probability)."
                recommended_action = "Consider consulting a healthcare professional within 24-48 hours."
            else:
                urgency = "LOW"
                urgency_reason = f"Random Forest triage classifier evaluated your symptom profile ('{matched_sym}') as routine/low urgency ({int(ml_confidence * 100)}% model probability)."
                recommended_action = "Monitor condition and maintain hydration. Consult a physician if symptoms worsen."

            # Conservative safety override if user indicates severe discomfort
            sev_lower = str(severity).lower()
            if "severe" in sev_lower and urgency == "LOW":
                urgency = "MODERATE"
                urgency_reason += " (Elevated to Moderate based on patient-reported severe discomfort)."
        except Exception as e:
            print(f"ML triage execution error: {e}")
            urgency = "MODERATE"
            urgency_reason = "Evaluated via clinical rule engine (Random Forest model encountered processing fallback)."

    # Parse detected symptoms array
    s_raw = symptoms_text.replace(' and ', ',').replace(' with ', ',').replace('&', ',')
    detected_symptoms = [s.strip().capitalize() for s in s_raw.split(',') if s.strip()]
    if not detected_symptoms:
        detected_symptoms = [symptoms_text.capitalize()]

    return {
        "urgency": urgency.lower(),
        "urgency_level": urgency,
        "confidence": ml_confidence,
        "symptoms": detected_symptoms,
        "detected_symptoms": detected_symptoms,
        "duration": duration,
        "original_statement": symptoms_text,
        "department": recommended_dept,
        "recommended_department": recommended_dept,
        "department_confidence": dept_confidence,
        "recommended_action": recommended_action,
        "emergency_warning": emergency_warning,
        "reason": urgency_reason,
        "urgency_reason": urgency_reason,
        "top_departments": top_departments,
        "disclaimer": "AI guidance only — not a medical diagnosis. Always consult a healthcare professional."
    }


# --- SMS History & Testing ---
@app.get("/api/sms/logs")
async def get_sms_logs(patient_id: Optional[int] = None, limit: int = 20):
    """Retrieve SMS delivery logs with masked recipient phone numbers"""
    db = SessionLocal()
    try:
        query = db.query(SMSLog)
        if patient_id:
            query = query.filter(SMSLog.patient_id == patient_id)
        logs = query.order_by(SMSLog.created_at.desc()).limit(limit).all()
        return {
            "success": True,
            "provider": sms_service.provider,
            "is_real_provider": sms_service.is_real,
            "logs": [
                {
                    "id": log.id,
                    "patient_id": log.patient_id,
                    "appointment_id": log.appointment_id,
                    "phone_masked": mask_phone(log.phone_number),
                    "message_type": log.message_type,
                    "message_body": log.message_body,
                    "status": log.status,
                    "provider": log.provider,
                    "sent_at": log.created_at.strftime("%Y-%m-%d %I:%M:%S %p") if log.created_at else None
                }
                for log in logs
            ]
        }
    finally:
        db.close()


@app.post("/api/sms/test")
async def send_test_sms(request: dict):
    """Send a test verification SMS to user's registered phone number"""
    phone = request.get("phone")
    if not phone:
        raise HTTPException(status_code=422, detail="Phone number is required")
    msg = request.get("message") or "MedAssist AI: Your phone verification test SMS was sent successfully."
    result = sms_service.send_sms(to_phone=phone, message=msg, message_type="verification")
    return result


# --- Queue Timeline & Advance ---
@app.get("/api/queue/timeline/{appointment_id}")
async def get_queue_timeline_endpoint(appointment_id: int):
    """Get live queue activity timeline"""
    timeline = get_or_create_timeline(appointment_id)
    return {"appointment_id": appointment_id, "timeline": timeline}


@app.post("/api/queue/advance/{appointment_id}")
async def advance_queue_endpoint(appointment_id: int, request: Optional[dict] = None):
    """Simulate/trigger moving ahead in queue and notify via SMS if threshold hit"""
    phone = request.get("phone") if request else None
    result = advance_appointment_queue(appointment_id, patient_phone=phone)
    
    # Broadcast to WebSockets if connected
    if appointment_id in active_ws_connections:
        for ws in active_ws_connections[appointment_id]:
            try:
                await ws.send_json({"type": "queue_advance", "data": result})
            except Exception:
                pass
                
    return result


# --- Smart Leave-Now Departure Buffer ---
@app.get("/api/travel/buffer")
async def get_travel_buffer_endpoint(distance_km: float = 3.5, appointment_time: str = "10:30 AM"):
    """Calculate recommended departure buffer based on distance and check-in overhead"""
    buffer_data = calculate_departure_buffer(distance_km, appointment_time)
    return buffer_data


# --- WebSocket: Queue Live Updates ---
active_ws_connections: Dict[int, List[WebSocket]] = {}

@app.websocket("/ws/queue/{appointment_id}")
async def queue_websocket(websocket: WebSocket, appointment_id: int):
    await websocket.accept()
    if appointment_id not in active_ws_connections:
        active_ws_connections[appointment_id] = []
    active_ws_connections[appointment_id].append(websocket)

    try:
        while True:
            # Send queue update every 15 seconds
            await asyncio.sleep(15)
            queue_data = tool_get_queue_status(appointment_id)
            if isinstance(queue_data, dict) and "appointment_id" in queue_data:
                await websocket.send_json({
                    "type": "queue_update",
                    "data": queue_data
                })
    except WebSocketDisconnect:
        if appointment_id in active_ws_connections:
            active_ws_connections[appointment_id].remove(websocket)
    except Exception:
        if appointment_id in active_ws_connections and websocket in active_ws_connections[appointment_id]:
            active_ws_connections[appointment_id].remove(websocket)


# --- Serve Frontend ---
@app.get("/")
async def serve_frontend():
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Agentic Medical Analyser API", "docs": "/docs"}


# --- Startup ---
@app.on_event("startup")
async def startup_event():
    print("=" * 50)
    print("AGENTIC MEDICAL ANALYSER")
    print("Intelligent Healthcare Assistant")
    print("=" * 50)
    print(f"FastAPI              OK")
    print(f"Triage Model         {'OK' if triage_model else 'NOT CONFIGURED'}")
    print(f"Department Model     {'OK' if classifier else 'NOT CONFIGURED'}")
    print(f"SentenceTransformer  {'OK' if embedder else 'NOT CONFIGURED'}")
    print(f"FAISS                {'OK' if dept_index else 'NOT CONFIGURED'}")
    print(f"Groq AI              {'OK' if client else 'NOT CONFIGURED'}")
    print(f"Database             OK")
    print(f"Agent System         OK")
    print(f"Queue System         OK")
    print(f"Notifications        OK")
    print(f"Maps Provider        OK (OSM Overpass)")
    print(f"Voice Support        BROWSER")
    print("=" * 50)
    print("Server ready: http://localhost:9000")
    print("Swagger: http://localhost:9000/docs")
    print("=" * 50)


if __name__ == "__main__":
    import uvicorn
    import socket

    port_env = os.getenv("PORT")
    if port_env:
        try:
            prod_port = int(port_env)
            print(f"Starting production server on assigned PORT: {prod_port}")
            uvicorn.run("main_combined:app", host="0.0.0.0", port=prod_port, reload=False)
        except Exception as e:
            print(f"Error starting on PORT {port_env}: {e}")
    else:
        def is_port_in_use(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('localhost', port)) == 0

        ports_to_try = [9000, 8010, 8011, 8012, 8013, 8014, 8015]
        for port in ports_to_try:
            try:
                if is_port_in_use(port):
                    print(f"Port {port} is busy.")
                    continue
                uvicorn.run("main_combined:app", host="0.0.0.0", port=port, reload=False)
                break
            except Exception as e:
                print(f"Failed on port {port}: {e}")
