# api/auth.py
import os
import functools
from flask import request, jsonify

# Load the expected API key from environment variables
# Ensure you set EXTERNAL_API_KEY in your .env file
EXPECTED_API_KEY = os.environ.get("EXTERNAL_API_KEY")

def require_api_key(f):
    """Decorator to require a valid API key in the X-API-Key header."""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        expected_key = os.environ.get('EXTERNAL_API_KEY')
        
        if not api_key:
            return jsonify({"error": "No API key provided"}), 401
            
        if api_key != expected_key:
            return jsonify({"error": "Invalid API key"}), 401
            
        return f(*args, **kwargs)
    return decorated_function