# api/auth.py
import os
from functools import wraps
from flask import request, jsonify

# Load the expected API key from environment variables
# Ensure you set EXTERNAL_API_KEY in your .env file
EXPECTED_API_KEY = os.environ.get("EXTERNAL_API_KEY")

def require_api_key(f):
    """Decorator to require a valid API key in the X-API-Key header."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not EXPECTED_API_KEY:
            # Server-side configuration error if the key isn't set
            print("ERROR: EXTERNAL_API_KEY environment variable not set.")
            return jsonify({"error": "Server configuration error: API Key not set."}), 500

        # Debug all headers
        print("Received headers:", dict(request.headers))
        
        provided_key = request.headers.get('X-API-Key')
        print(f"Provided API key: {provided_key}")
        print(f"Expected API key: {EXPECTED_API_KEY}")

        if not provided_key:
            return jsonify({"error": "Unauthorized: Missing X-API-Key header."}), 401
        
        if provided_key != EXPECTED_API_KEY:
            return jsonify({"error": "Unauthorized: Invalid API Key."}), 401
        
        # If keys match, proceed with the original route function
        return f(*args, **kwargs)
    return decorated_function