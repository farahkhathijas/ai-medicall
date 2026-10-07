import json
import math
import requests
from sqlalchemy.orm import Session

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_coordinates_for_city(city_name: str):
    """Uses OSM Nominatim to geocode a city name into lat/lon"""
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

def search_hospitals(department: str = None, latitude: float = None, longitude: float = None, radius_km: float = 10.0, query: str = None):
    # If text query provided instead of lat/lon, attempt geocoding
    if query and (latitude is None or longitude is None):
        latitude, longitude = get_coordinates_for_city(query)

    # Without coordinates, we cannot reliably find nearby real hospitals via OSM
    if latitude is None or longitude is None:
        return {"success": False, "error": "Location required for real hospital search."}

    radius_m = radius_km * 1000

    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json][timeout:25];
    (
      node["amenity"="hospital"](around:{radius_m},{latitude},{longitude});
      way["amenity"="hospital"](around:{radius_m},{latitude},{longitude});
      relation["amenity"="hospital"](around:{radius_m},{latitude},{longitude});
    );
    out center;
    """
    
    results = []
    try:
        response = requests.post(overpass_url, data={'data': overpass_query}, timeout=30)
        data = response.json()
        
        for element in data.get('elements', []):
            tags = element.get('tags', {})
            
            name = tags.get('name')
            if not name:
                continue
                
            lat = element.get('lat') or (element.get('center', {})).get('lat')
            lon = element.get('lon') or (element.get('center', {})).get('lon')
            
            if not lat or not lon:
                continue

            address = tags.get('addr:street', '') + " " + tags.get('addr:city', '')
            if not address.strip():
                address = "Address unavailable"

            phone = tags.get('phone') or tags.get('contact:phone')
            website = tags.get('website') or tags.get('contact:website')
            
            # Since OSM doesn't natively categorize hospital departments uniformly, 
            # we provide a generic list or attempt to extract from healthcare:speciality
            speciality = tags.get('healthcare:speciality', department or 'General Medicine')
            deps = [d.strip().title() for d in speciality.split(';')]

            dist = haversine(latitude, longitude, lat, lon)
            
            results.append({
                "id": element.get('id'),
                "name": name,
                "latitude": lat,
                "longitude": lon,
                "address": address.strip(),
                "phone": phone,
                "website": website,
                "departments": deps,
                "distance_km": round(dist, 2)
            })
            
        results.sort(key=lambda x: x["distance_km"])
        
        # Limit to top 20 to avoid payload bloat
        return {"success": True, "hospitals": results[:20]}
        
    except Exception as e:
        print(f"Overpass API error: {e}")
        return {"success": False, "error": "External hospital provider (OSM) unavailable."}


def _get_db_session():
    from main_combined import SessionLocal
    return SessionLocal()

def search_doctors(department: str = None, hospital_id: int = None):
    from main_combined import Doctor, Hospital
    db = _get_db_session()
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
            hospital_name = hospital.name if hospital else "Unknown Hospital"
            result.append({
                "id": d.id,
                "hospital_id": d.hospital_id,
                "name": d.name,
                "department": d.department,
                "hospital": hospital_name
            })
        return result
    finally:
        db.close()

def get_available_slots(doctor_id: int):
    from main_combined import AppointmentSlot
    db = _get_db_session()
    try:
        slots = db.query(AppointmentSlot).filter(
            AppointmentSlot.doctor_id == doctor_id,
            AppointmentSlot.available == True
        ).all()
        return [{"id": s.id, "date": s.date, "start_time": s.start_time, "end_time": s.end_time} for s in slots]
    finally:
        db.close()

def create_appointment(patient_id: int, doctor_id: int, hospital_id: int, slot_id: int):
    from main_combined import Appointment, AppointmentSlot
    db = _get_db_session()
    try:
        slot = db.query(AppointmentSlot).filter(AppointmentSlot.id == slot_id).first()
        if not slot or not slot.available:
            return {"error": "Slot unavailable."}
        
        slot.available = False
        new_apt = Appointment(
            patient_id=patient_id,
            doctor_id=doctor_id,
            hospital_id=hospital_id,
            slot_id=slot_id,
            status="Confirmed"
        )
        db.add(new_apt)
        db.commit()
        db.refresh(new_apt)
        return {"id": new_apt.id, "status": new_apt.status}
    except Exception as e:
        db.rollback()
        return {"error": str(e)}
    finally:
        db.close()

def get_appointments(patient_id: int):
    from main_combined import Appointment, Doctor, Hospital, AppointmentSlot
    db = _get_db_session()
    try:
        appointments = db.query(Appointment).filter(Appointment.patient_id == patient_id).all()
        result = []
        for apt in appointments:
            doc = db.query(Doctor).filter(Doctor.id == apt.doctor_id).first()
            hosp = db.query(Hospital).filter(Hospital.id == apt.hospital_id).first()
            slot = db.query(AppointmentSlot).filter(AppointmentSlot.id == apt.slot_id).first()
            
            result.append({
                "id": apt.id,
                "doctor": doc.name if doc else "Unknown",
                "hospital": hosp.name if hosp else "Unknown",
                "date": slot.date if slot else "Unknown",
                "time": slot.start_time if slot else "Unknown",
                "status": apt.status
            })
        return result
    finally:
        db.close()

def cancel_appointment(appointment_id: int):
    from main_combined import Appointment, AppointmentSlot
    db = _get_db_session()
    try:
        apt = db.query(Appointment).filter(Appointment.id == appointment_id).first()
        if not apt:
            return {"error": "Appointment not found"}
        
        if apt.status != "Cancelled":
            apt.status = "Cancelled"
            slot = db.query(AppointmentSlot).filter(AppointmentSlot.id == apt.slot_id).first()
            if slot:
                slot.available = True
            db.commit()
            return {"success": True, "message": "Cancelled"}
        return {"error": "Already cancelled"}
    except Exception as e:
        db.rollback()
        return {"error": str(e)}
    finally:
        db.close()

