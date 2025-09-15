from geopy.distance import geodesic
from twilio.rest import Client
from datetime import datetime
import streamlit as st
import requests

account_sid = '**********************'
auth_token = '**********************'
messaging_service_sid = '*****************"
recipient_number = '***********'

client = Client(account_sid, auth_token)
camera_locations = {
    "Camera 1": {"latitude": 28.6129, "longitude": 77.2295, "location": "India Gate, New Delhi"},
    "Camera 2": {"latitude": 40.7128, "longitude": -74.0060, "location": "Downtown Square"}
}

def get_nearest_emergency_services2(lat, lon, service_type):
    """
    Returns nearest hospital/police station with contact number
    """
    # Define OSM query filters
    tags = {
        "hospital": ['"amenity"="hospital"', '"emergency"="yes"'],
        "police": ['"amenity"="police"']
    }

    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    (
      node[{tags[service_type][0]}](around:5000,{lat},{lon});
      way[{tags[service_type][0]}](around:5000,{lat},{lon});
    );
    out center;
    """
    
    try:
        response = requests.post(overpass_url, data=query)
        response.raise_for_status()
        results = response.json().get('elements', [])
    except Exception as e:
        st.error(f"Error fetching {service_type}: {e}")
        return None

    nearest = None
    min_distance = float('inf')

    for element in results:
        # Get location
        if 'center' in element:
            elat, elon = element['center']['lat'], element['center']['lon']
        else:
            elat, elon = element['lat'], element['lon']

        # Get contact number
        contacts = {
            'phone': element['tags'].get('phone'),
            'emergency_phone': element['tags'].get('emergency_phone')
        }
        phone = contacts['emergency_phone'] or contacts['phone']

        # Calculate distance
        distance = geodesic((lat, lon), (elat, elon)).km

        if phone and distance < min_distance:
            min_distance = distance
            nearest = {
                'name': element['tags'].get('name', 'Unknown'),
                'phone': phone,
                'distance': round(distance, 2)
            }

    return nearest


def send_alert2(camera_id, timestamp, coords):
    google_maps_link = f"https://www.google.com/maps?q={coords['latitude']},{coords['longitude']}"
    # Fetch nearest emergency services
    hospital = get_nearest_emergency_services2(coords['latitude'], coords['longitude'], "hospital")
    police = get_nearest_emergency_services2(coords['latitude'], coords['longitude'], "police")

    alert_message = (
        f"ALERT:Road accident \n"
        # f"Camera: {camera_id}\n"
        # f"Location: {coords['location']} (Latitude {coords['latitude']}, Longitude {coords['longitude']})\n"
        f"Time: {timestamp}\n"
        f"Google Maps: {google_maps_link}\n"
    )

    # Add emergency services info
    if hospital:
        alert_message += f"Nearest Hospital: {hospital['name']}, Phone: {hospital['phone']}, Distance: {hospital['distance']}km\n"
    if police:
        alert_message += f"Nearest Police Station: {police['name']}, Phone: {police['phone']}, Distance: {police['distance']}km\n"

    alert_message += "Emergency response required!"

    try:
        message = client.messages.create(
            body=alert_message,
            messaging_service_sid=messaging_service_sid,
            to=recipient_number
        )
        st.sidebar.success(f"Notification sent successfully! SID: {message.sid}")
    except Exception as e:
        st.sidebar.error(f"Failed to send notification: {e}")
    

    return hospital, police  # Return data for Streamlit display
