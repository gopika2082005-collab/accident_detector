import cv2
from playsound import playsound
import smtplib
from email.message import EmailMessage
import time
import os
import threading
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def play_alarm():
    if os.path.exists("alarm.wav"):
        try:
            # block=False prevents the code from freezing while the alarm plays
            playsound("alarm.wav", block=False)
        except Exception:
            pass

def save_frame(frame):
    filename = f"accident_{int(time.time())}.jpg"
    cv2.imwrite(filename, frame)
    return filename

def get_location():
    """Fetch location based on IP address."""
    try:
        response = requests.get("https://ipapi.co/json/", timeout=5)
        data = response.json()
        lat, lon = data.get("latitude"), data.get("longitude")
        city, region = data.get("city"), data.get("region")
        
        if lat and lon:
            return lat, lon, f"{city}, {region}"
    except Exception as e:
        print(f"Location fetching failed: {e}")
    
    # Fallback to defaults from env or simulated location
    lat = os.getenv("DEFAULT_LATITUDE", "12.9716")
    lon = os.getenv("DEFAULT_LONGITUDE", "77.5946")
    return lat, lon, "Unknown (Fallback Location)"

def _send_email_task(image_path):
    # Retrieve credentials from .env, or use the old defaults if missing so the MVP doesn't crash
    SENDER_EMAIL = os.getenv("SENDER_EMAIL", "gopika2082005@gmail.com")
    APP_PASSWORD = os.getenv("APP_PASSWORD", "qwsa patg wxdn mzvk")
    # Read receiver emails, split by comma if multiple
    receivers_str = os.getenv("RECEIVER_EMAILS", "gopika2585@gmail.com")
    
    # Process recipients list
    recipients = [email.strip() for email in receivers_str.split(",") if email.strip()]

    # Fetch exact location
    lat, lon, address = get_location()
    maps_link = f"https://www.google.com/maps?q={lat},{lon}"

    msg = EmailMessage()
    msg['Subject'] = "🚨 URGENT: Accident Detected at " + address
    msg['From'] = SENDER_EMAIL
    msg['To'] = ", ".join(recipients)
    
    body_text = f"An accident has been detected by the AI system.\n\n"
    body_text += f"Location: {address}\n"
    body_text += f"Exact Coordinates: {lat}, {lon}\n"
    body_text += f"Google Maps: {maps_link}\n\n"
    body_text += "Please review the attached evidence image and dispatch emergency services (Police/Hospital)."
    
    msg.set_content(body_text)

    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
            msg.add_attachment(
                image_data,
                maintype='image',
                subtype='jpeg',
                filename=os.path.basename(image_path)
            )
    except Exception as e:
        print("Could not read image for email attachment:", e)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, APP_PASSWORD)
            smtp.send_message(msg)
        print("✅ Alert Email successfully sent to Hospitals/Police Stations:", ", ".join(recipients))
    except Exception as e:
        print("❌ Email failed:", e)

def send_email(image_path):
    """Starts the send_email task in a background thread to prevent camera stuttering."""
    thread = threading.Thread(target=_send_email_task, args=(image_path,))
    thread.daemon = True # Allows program to exit even if thread is running
    thread.start()