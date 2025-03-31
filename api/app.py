import os
import json
import uuid
import time
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from browser_use import Agent
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Keep track of browser tasks
tasks = {}

def get_model_instance(model_name):
    """Get the appropriate LLM instance based on model name"""
    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        raise ValueError("OpenAI API key not configured. Please set OPENAI_API_KEY in the .env file.")
    
    # Define supported models and their configurations
    models = {
        "gpt-4o": ChatOpenAI(model="gpt-4o"),
        "gpt-4-turbo": ChatOpenAI(model="gpt-4-turbo"),
        "gpt-4": ChatOpenAI(model="gpt-4"),
        "gpt-3.5-turbo": ChatOpenAI(model="gpt-3.5-turbo"),
    }
    
    return models.get(model_name, models["gpt-4o"])  # Default to gpt-4o if model not found

async def run_browser_task(task_id, task_description, model_name):
    """Run a browser task asynchronously"""
    try:
        # Update task status to running
        tasks[task_id]["status"] = "running"
        tasks[task_id]["updated_at"] = datetime.now().isoformat()
        
        # Initialize the agent with the specified LLM
        llm = get_model_instance(model_name)
        agent = Agent(
            task=task_description,
            llm=llm,
        )
        
        # Run the agent
        result = await agent.run()
        
        # Update task with result
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = result
        tasks[task_id]["updated_at"] = datetime.now().isoformat()
    
    except Exception as e:
        # Update task with error
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["updated_at"] = datetime.now().isoformat()

@app.route("/api/browser-tasks", methods=["POST"])
def create_browser_task():
    """Create a new browser task"""
    try:
        data = request.json
        task_description = data.get("task")
        model_name = data.get("model", "gpt-4o")
        
        if not task_description:
            return jsonify({"error": "Task description is required"}), 400
        
        # Generate a unique task ID
        task_id = str(uuid.uuid4())
        
        # Create task entry
        tasks[task_id] = {
            "id": task_id,
            "task": task_description,
            "model": model_name,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        
        # Run the task in the background
        def run_task():
            asyncio.run(run_browser_task(task_id, task_description, model_name))
        
        # Start task in a separate thread to allow API to respond immediately
        import threading
        thread = threading.Thread(target=run_task)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "id": task_id,
            "status": "pending",
            "message": "Task created successfully"
        }), 201
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/browser-tasks", methods=["GET"])
def list_browser_tasks():
    """List all browser tasks"""
    return jsonify(list(tasks.values())), 200

@app.route("/api/browser-tasks/<task_id>", methods=["GET"])
def get_browser_task(task_id):
    """Get the status and result of a browser task"""
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    
    return jsonify(task), 200

@app.route("/api/supported-models", methods=["GET"])
def get_supported_models():
    """Get list of supported LLM models"""
    models = [
        {"id": "gpt-4o", "name": "GPT-4o"},
        {"id": "gpt-4-turbo", "name": "GPT-4 Turbo"},
        {"id": "gpt-4", "name": "GPT-4"},
        {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo"},
    ]
    return jsonify(models), 200

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    # Check if API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    api_key_status = "available" if api_key and api_key != "your_openai_api_key_here" else "missing"
    
    # Check if Playwright is installed
    try:
        import playwright
        playwright_status = "installed"
    except ImportError:
        playwright_status = "missing"
    
    # Return health status
    return jsonify({
        "status": "healthy",
        "openai_api_key": api_key_status,
        "playwright": playwright_status,
        "timestamp": datetime.now().isoformat()
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)