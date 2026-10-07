"""
Live Queue Tracking & Intelligence Service
Handles real-time position updates, WebSocket broadcasting, timeline logging, and smart travel buffer calculation.
"""

import math
import datetime
from typing import Dict, Any, List, Optional
from services.sms_service import sms_service

# In-memory queue state with timeline logs per appointment
_queue_timelines: Dict[int, List[Dict[str, Any]]] = {}
_last_sms_thresholds: Dict[int, int] = {}

def get_or_create_timeline(appointment_id: int, initial_pos: int = 5, wait_min: int = 25) -> List[Dict[str, Any]]:
    """Initialize or retrieve activity timeline for an appointment"""
    if appointment_id not in _queue_timelines:
        now = datetime.datetime.now()
        t1 = (now - datetime.timedelta(minutes=15)).strftime("%I:%M %p")
        t2 = (now - datetime.timedelta(minutes=10)).strftime("%I:%M %p")
        
        _queue_timelines[appointment_id] = [
            {"time": t1, "event": "Appointment Confirmed", "icon": "✅", "status": "completed"},
            {"time": t2, "event": "Checked In at Clinic Reception", "icon": "🏥", "status": "completed"},
            {"time": now.strftime("%I:%M %p"), "event": f"Queue Position #{initial_pos} Assigned ({initial_pos-1} ahead)", "icon": "📊", "status": "current"}
        ]
    return _queue_timelines[appointment_id]

def add_timeline_event(appointment_id: int, event_text: str, icon: str = "📌", status: str = "completed"):
    """Add a new timestamped event to the queue timeline"""
    timeline = get_or_create_timeline(appointment_id)
    now_str = datetime.datetime.now().strftime("%I:%M %p")
    # Mark previous as completed
    for item in timeline:
        if item.get("status") == "current":
            item["status"] = "completed"
    timeline.append({
        "time": now_str,
        "event": event_text,
        "icon": icon,
        "status": status
    })

def calculate_departure_buffer(distance_km: float, appointment_time_str: str) -> Dict[str, Any]:
    """
    Calculates Smart Leave-Now departure recommendation.
    Accounts for distance, 3.5 min/km urban travel rate, and a 12 min check-in registration buffer.
    """
    if not distance_km or distance_km <= 0:
        distance_km = 3.5
    
    travel_time_min = max(5, int(distance_km * 3.5))
    clinic_buffer_min = 12
    total_lead_time_min = travel_time_min + clinic_buffer_min

    # Try parsing appointment time like "10:30 AM" or "02:00 PM"
    departure_time_str = "25 minutes before your slot"
    try:
        now = datetime.datetime.now()
        # Parse time string
        apt_time = datetime.datetime.strptime(appointment_time_str.strip(), "%I:%M %p")
        apt_dt = now.replace(hour=apt_time.hour, minute=apt_time.minute, second=0, microsecond=0)
        dep_dt = apt_dt - datetime.timedelta(minutes=total_lead_time_min)
        departure_time_str = dep_dt.strftime("%I:%M %p")
    except Exception:
        pass

    return {
        "travel_time_min": travel_time_min,
        "clinic_buffer_min": clinic_buffer_min,
        "total_lead_time_min": total_lead_time_min,
        "recommended_departure_time": departure_time_str,
        "message": f"Based on ~{travel_time_min} mins travel ({distance_km:.1f} km) and a {clinic_buffer_min}-min check-in buffer, leaving around {departure_time_str} is recommended."
    }

def advance_appointment_queue(appointment_id: int, patient_phone: Optional[str] = None) -> Dict[str, Any]:
    """
    Advances queue position by 1 step, triggers SMS at key thresholds (5, 2, 1), and appends timeline.
    """
    from main_combined import SessionLocal, Appointment, QueueEntry, Notification
    db = SessionLocal()
    try:
        apt = db.query(Appointment).filter(Appointment.id == appointment_id).first()
        if not apt:
            return {"error": "Appointment not found"}

        q_entry = db.query(QueueEntry).filter(QueueEntry.appointment_id == appointment_id).first()
        if not q_entry:
            q_entry = QueueEntry(
                appointment_id=appointment_id,
                doctor_id=apt.doctor_id,
                queue_number=apt.queue_number or 105,
                position=apt.queue_position or 5,
                status="waiting",
                estimated_wait_minutes=apt.estimated_wait_minutes or 20
            )
            db.add(q_entry)
            db.commit()
            db.refresh(q_entry)

        # Advance position
        if q_entry.position > 1:
            q_entry.position -= 1
            q_entry.estimated_wait_minutes = max(3, q_entry.position * 5)
            q_entry.status = "waiting" if q_entry.position > 1 else "your_turn"
        else:
            q_entry.position = 0
            q_entry.estimated_wait_minutes = 0
            q_entry.status = "in_consultation"

        apt.queue_position = q_entry.position
        apt.estimated_wait_minutes = q_entry.estimated_wait_minutes
        db.commit()

        pos = q_entry.position
        people_ahead = max(0, pos - 1)
        wait_min = q_entry.estimated_wait_minutes

        # Timeline update
        if pos == 0:
            add_timeline_event(appointment_id, "In Consultation with Doctor", "👨‍⚕️", "current")
        elif pos == 1:
            add_timeline_event(appointment_id, "Called for Consultation (Your Turn)", "🎉", "current")
        elif pos == 2:
            add_timeline_event(appointment_id, f"Queue Position #{pos} — Almost your turn (1 person ahead)", "⏰", "current")
        else:
            add_timeline_event(appointment_id, f"Queue Position #{pos} ({people_ahead} people ahead, ~{wait_min} min)", "📊", "current")

        # Check notification thresholds & send SMS
        sms_result = None
        last_sent = _last_sms_thresholds.get(appointment_id, 999)
        if patient_phone and pos != last_sent:
            if pos == 1 or pos == 2 or (pos == 5 and last_sent > 5):
                sms_result = sms_service.send_queue_update(
                    phone=patient_phone,
                    position=pos,
                    people_ahead=people_ahead,
                    est_wait_min=wait_min,
                    appointment_id=appointment_id,
                    patient_id=apt.patient_id
                )
                _last_sms_thresholds[appointment_id] = pos

        # In-app notification
        notif_msg = f"Queue update: You are now #{pos} ({people_ahead} ahead). Est. wait: ~{wait_min} mins."
        if pos <= 1:
            notif_msg = f"🎉 It's your turn! Please proceed to the consultation room."
        notif = Notification(
            patient_id=apt.patient_id,
            appointment_id=appointment_id,
            message=notif_msg,
            notification_type="queue_update"
        )
        db.add(notif)
        db.commit()

        return {
            "appointment_id": appointment_id,
            "queue_number": q_entry.queue_number,
            "position": pos,
            "people_ahead": people_ahead,
            "estimated_wait_minutes": wait_min,
            "status": q_entry.status,
            "timeline": get_or_create_timeline(appointment_id),
            "sms_sent": bool(sms_result and sms_result.get("success")),
            "sms_status": sms_result.get("status") if sms_result else "NONE"
        }
    finally:
        db.close()
