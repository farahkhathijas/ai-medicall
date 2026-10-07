"""
SMS & Notification Delivery Service
Supports Twilio, MSG91, and a verified Simulation/Console provider with database logging.
"""

import os
import json
import re
import datetime
import requests
from typing import Optional, Dict, Any, List

def mask_phone(phone: Optional[str]) -> str:
    """Masks a phone number for privacy (e.g. +91 9876543210 -> +91 ******3210)"""
    if not phone:
        return "Not provided"
    cleaned = re.sub(r'[\s\-()]', '', phone)
    if len(cleaned) <= 4:
        return cleaned
    return cleaned[:3] + "******" + cleaned[-4:]

class SMSService:
    def __init__(self):
        self.provider = os.getenv("SMS_PROVIDER", "").lower().strip()
        # Twilio Config
        self.twilio_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
        self.twilio_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
        self.twilio_phone = os.getenv("TWILIO_PHONE_NUMBER", "").strip()
        
        # MSG91 Config
        self.msg91_auth_key = os.getenv("MSG91_AUTH_KEY", "").strip()
        self.msg91_sender_id = os.getenv("MSG91_SENDER_ID", "MEDAST").strip()

        # Fast2SMS Config
        self.fast2sms_key = os.getenv("FAST2SMS_API_KEY", "").strip()
        
        # Determine effective mode
        self.is_real = False
        if self.twilio_sid and self.twilio_token and not self.twilio_sid.startswith("your_") and not self.twilio_sid.startswith("placeholder"):
            self.is_real = True
            self.provider = "twilio"
        elif self.msg91_auth_key and not self.msg91_auth_key.startswith("your_") and not self.msg91_auth_key.startswith("placeholder"):
            self.is_real = True
            self.provider = "msg91"
        elif self.fast2sms_key and not self.fast2sms_key.startswith("your_") and not self.fast2sms_key.startswith("placeholder"):
            self.is_real = True
            self.provider = "fast2sms"
        elif self.provider == "simulation" or self.provider == "demo":
            self.is_real = False
            self.provider = "simulation"
        else:
            self.is_real = False
            self.provider = "none"
            
        print(f"SMS Service initialized: mode={'REAL' if self.is_real else 'UNCONFIGURED/DEMO'}, provider={self.provider}")

    def _log_sms_to_db(self, phone: str, message: str, message_type: str, status: str, provider: str, 
                       patient_id: Optional[int] = None, appointment_id: Optional[int] = None, 
                       provider_response_id: Optional[str] = None, error_message: Optional[str] = None):
        """Persist SMS delivery log into SQLite database"""
        try:
            from main_combined import SessionLocal, SMSLog
            db = SessionLocal()
            log_entry = SMSLog(
                patient_id=patient_id,
                appointment_id=appointment_id,
                phone_number=phone,
                message_type=message_type,
                message_body=message,
                status=status,
                provider=provider,
                provider_response_id=provider_response_id,
                error_message=error_message
            )
            db.add(log_entry)
            db.commit()
            db.refresh(log_entry)
            log_id = log_entry.id
            db.close()
            return log_id
        except Exception as e:
            print(f"Failed to log SMS to database: {e}")
            return None

    def send_sms(self, to_phone: str, message: str, message_type: str = "general", 
                 patient_id: Optional[int] = None, appointment_id: Optional[int] = None,
                 demo_mode: bool = False) -> Dict[str, Any]:
        """Send SMS via configured provider with truthful error reporting and database audit"""
        if not to_phone or len(to_phone.strip()) < 5:
            return {
                "success": False, 
                "error": "Invalid phone number provided", 
                "status": "FAILED",
                "provider": self.provider
            }

        to_phone = to_phone.strip()
        status = "FAILED"
        provider_resp_id = None
        error_msg = None

        if self.is_real and self.provider == "twilio":
            try:
                url = f"https://api.twilio.com/2010-04-01/Accounts/{self.twilio_sid}/Messages.json"
                resp = requests.post(
                    url,
                    data={"To": to_phone, "From": self.twilio_phone, "Body": message},
                    auth=(self.twilio_sid, self.twilio_token),
                    timeout=10
                )
                res_data = resp.json()
                if resp.status_code in [200, 201]:
                    provider_resp_id = res_data.get("sid")
                    status = "DELIVERED" if res_data.get("status") in ["delivered", "sent", "queued"] else "SENT"
                else:
                    status = "FAILED"
                    error_msg = res_data.get("message", f"Twilio HTTP error {resp.status_code}")
            except Exception as e:
                status = "FAILED"
                error_msg = f"Twilio connection error: {str(e)}"

        elif self.is_real and self.provider == "msg91":
            try:
                url = "https://api.msg91.com/api/v5/flow/"
                headers = {"authkey": self.msg91_auth_key, "content-type": "application/json"}
                payload = {
                    "sender": self.msg91_sender_id,
                    "mobiles": re.sub(r'[^0-9]', '', to_phone),
                    "message": message
                }
                resp = requests.post(url, json=payload, headers=headers, timeout=10)
                res_data = resp.json()
                if resp.status_code == 200 and res_data.get("type") == "success":
                    provider_resp_id = str(res_data.get("message"))
                    status = "SENT"
                else:
                    status = "FAILED"
                    error_msg = res_data.get("message", f"MSG91 error {resp.status_code}")
            except Exception as e:
                status = "FAILED"
                error_msg = f"MSG91 connection error: {str(e)}"

        elif self.is_real and self.provider == "fast2sms":
            try:
                url = "https://www.fast2sms.com/dev/bulkV2"
                clean_phone = re.sub(r'[^0-9]', '', to_phone)[-10:]
                payload = {
                    "authorization": self.fast2sms_key,
                    "route": "q",
                    "message": message,
                    "language": "english",
                    "flash": 0,
                    "numbers": clean_phone
                }
                resp = requests.post(url, data=payload, timeout=10)
                res_data = resp.json()
                if resp.status_code == 200 and res_data.get("return") is True:
                    provider_resp_id = res_data.get("request_id")
                    status = "SENT"
                else:
                    status = "FAILED"
                    error_msg = res_data.get("message", f"Fast2SMS error {resp.status_code}")
            except Exception as e:
                status = "FAILED"
                error_msg = f"Fast2SMS connection error: {str(e)}"

        elif demo_mode or self.provider == "simulation":
            # Explicit Demo Mode Only
            status = "SIMULATED"
            provider_resp_id = f"sim_{int(datetime.datetime.now().timestamp() * 1000)}"
            error_msg = "Demo mode simulation (No live network SMS dispatched)"

        else:
            # Unconfigured credentials: Truthful reporting
            status = "NOT_CONFIGURED"
            error_msg = "SMS credentials not configured in .env. Configure TWILIO_ACCOUNT_SID, MSG91_AUTH_KEY, or FAST2SMS_API_KEY."
            provider_resp_id = None

        # Log to Database
        log_id = self._log_sms_to_db(
            phone=to_phone,
            message=message,
            message_type=message_type,
            status=status,
            provider=self.provider if self.is_real else ("simulation" if (demo_mode or self.provider=="simulation") else "none"),
            patient_id=patient_id,
            appointment_id=appointment_id,
            provider_response_id=provider_resp_id,
            error_message=error_msg
        )

        return {
            "success": status in ["SENT", "DELIVERED", "SIMULATED"],
            "status": status,
            "provider": self.provider if self.is_real else ("simulation" if (demo_mode or self.provider=="simulation") else "none"),
            "message_id": provider_resp_id or (f"log_{log_id}" if log_id else None),
            "recipient_masked": mask_phone(to_phone),
            "log_id": log_id,
            "error": error_msg
        }

    def send_appointment_confirmation(self, phone: str, hospital: str, doctor: str, department: str,
                                      date_str: str, time_str: str, appointment_id: int, queue_number: int,
                                      patient_id: Optional[int] = None) -> Dict[str, Any]:
        """Send appointment booking confirmation SMS"""
        body = (
            f"✅ Appointment Confirmed!\n"
            f"Hospital: {hospital}\n"
            f"Department: {department}\n"
            f"Doctor: {doctor}\n"
            f"Date & Time: {date_str} at {time_str}\n"
            f"Appointment ID: #{appointment_id}\n"
            f"Queue Token: #{queue_number}\n\n"
            f"Track your live queue & buffer time in the MedAssist AI app."
        )
        return self.send_sms(phone, body, message_type="appointment_confirmed", 
                             patient_id=patient_id, appointment_id=appointment_id)

    def send_queue_update(self, phone: str, position: int, people_ahead: int, est_wait_min: int,
                          appointment_id: Optional[int] = None, patient_id: Optional[int] = None) -> Dict[str, Any]:
        """Send queue position update SMS"""
        if position <= 1 or people_ahead == 0:
            body = (
                f"🚨 Your Turn Now!\n"
                f"Appointment #{appointment_id or ''}\n"
                f"Please proceed directly to the doctor consultation room.\n"
                f"Your token is being called."
            )
            msg_type = "turn_reached"
        elif position <= 2:
            body = (
                f"⏰ Appointment Approaching!\n"
                f"You are currently #{position} in the queue ({people_ahead} person ahead).\n"
                f"Estimated wait: ~{est_wait_min} minutes. Please remain near the consultation area."
            )
            msg_type = "turn_approaching"
        else:
            body = (
                f"📊 Queue Update:\n"
                f"You are now #{position} in queue with {people_ahead} people ahead.\n"
                f"Estimated wait time: ~{est_wait_min} minutes."
            )
            msg_type = "queue_update"

        return self.send_sms(phone, body, message_type=msg_type, 
                             patient_id=patient_id, appointment_id=appointment_id)


# Global singleton instance
sms_service = SMSService()
