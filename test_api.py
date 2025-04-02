#!/usr/bin/env python
import os
import flask
from flask import Flask, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Simple logging
app.logger.setLevel("INFO")
print("Starting test API...")

# Configure Playwright in Replit environment (with fixed indentation)
if os.environ.get('REPL_ID'):
    app.logger.info("Replit environment detected, configuring resource limits...")
    # Add ulimit settings to increase file descriptor and process limits
    try:
        import resource
        # Try to increase the file descriptor limit
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        app.logger.info(f"File descriptor limit set to {hard}")
        
        # Try to increase the process limit
        soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
        resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
        app.logger.info(f"Process limit set to {hard}")
    except Exception as e:
        app.logger.warning(f"Could not increase resource limits: {str(e)}")

@app.route("/api/health", methods=["GET"])
def health_check():
    """Basic health check endpoint. Returns simple status."""
    return jsonify({
        "status": "ok",
        "message": "API is running"
    })

# Run the Flask app when script is executed directly
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)