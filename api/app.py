import os
import json
import uuid
import time
import asyncio
import subprocess
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

# Ensure Playwright browsers are installed
try:
    print("Installing Playwright browsers...")
    # Install just Firefox as it might have fewer dependencies
    subprocess.run(["python", "-m", "playwright", "install", "firefox"], check=True)
    print("Playwright Firefox browser installed successfully.")
except Exception as e:
    print(f"Error installing Playwright browsers: {e}")
    print("Will try to use the system-installed Firefox browser.")

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
        
        # Log agent initialization
        print(f"Initializing Agent for task: {task_id}")
        
        # Try different approaches to initialize the Agent
        browser_errors = []
        
        # First attempt - try with Firefox (system installed)
        try:
            print(f"Attempt 1: Using Firefox with no-sandbox for task {task_id}")
            agent = Agent(
                task=task_description,
                llm=llm,
                headless=True,
                browser_type="firefox",
                launch_args=["--no-sandbox"],
            )
            
            # Run the agent
            print(f"Running agent for task: {task_id}")
            result = await agent.run()
            
            # Success! Update task with result
            tasks[task_id]["status"] = "completed"
            tasks[task_id]["result"] = result
            tasks[task_id]["updated_at"] = datetime.now().isoformat()
            return  # Exit the function successfully
            
        except Exception as error1:
            print(f"Attempt 1 failed for task {task_id}: {str(error1)}")
            browser_errors.append(f"Firefox attempt failed: {str(error1)}")
        
        # Second attempt - try with Chromium
        try:
            print(f"Attempt 2: Using Chromium with no-sandbox for task {task_id}")
            agent = Agent(
                task=task_description,
                llm=llm,
                headless=True,
                browser_type="chromium",
                launch_args=["--no-sandbox", "--disable-gpu"],
            )
            
            # Run the agent
            print(f"Running agent (attempt 2) for task: {task_id}")
            result = await agent.run()
            
            # Success! Update task with result
            tasks[task_id]["status"] = "completed"
            tasks[task_id]["result"] = result
            tasks[task_id]["updated_at"] = datetime.now().isoformat()
            return  # Exit the function successfully
            
        except Exception as error2:
            print(f"Attempt 2 failed for task {task_id}: {str(error2)}")
            browser_errors.append(f"Chromium attempt failed: {str(error2)}")
        
        # If we got here, all attempts failed
        detailed_error = (
            f"Browser automation failed. Tried multiple approaches:\n"
            f"{browser_errors[0]}\n"
            f"{browser_errors[1] if len(browser_errors) > 1 else ''}\n\n"
            "This is likely due to missing system dependencies required by Playwright. "
            "The application requires a complete browser environment to function."
        )
        
        print(f"All browser initialization attempts failed for task {task_id}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = detailed_error
        tasks[task_id]["updated_at"] = datetime.now().isoformat()
    
    except Exception as e:
        # Handle general errors
        print(f"General error in task {task_id}: {str(e)}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = f"Failed to execute task: {str(e)}"
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
    health_data = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": {}
    }
    
    # Check OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    health_data["environment"]["openai_api_key"] = "available" if api_key and api_key != "your_openai_api_key_here" else "missing"
    
    # Check Playwright installation
    try:
        import playwright
        health_data["environment"]["playwright"] = {
            "status": "installed",
            "version": getattr(playwright, "__version__", "unknown")
        }
    except ImportError:
        health_data["environment"]["playwright"] = {
            "status": "missing"
        }
    
    # Check browser-use installation
    try:
        import browser_use
        health_data["environment"]["browser_use"] = {
            "status": "installed",
            "version": getattr(browser_use, "__version__", "unknown")
        }
    except ImportError:
        health_data["environment"]["browser_use"] = {
            "status": "missing"
        }
    
    # Get Python version
    import sys
    health_data["environment"]["python"] = {
        "version": sys.version,
        "executable": sys.executable
    }
    
    # Check for system dependencies common to Playwright
    system_deps = {}
    for lib in ["libnss3", "libxrandr2", "libgbm1", "libxshmfence1", "libdrm2"]:
        try:
            subprocess.run(["ldconfig", "-p"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            system_deps[lib] = "available"
        except:
            system_deps[lib] = "unknown"
    
    health_data["environment"]["system_dependencies"] = system_deps
    
    # Overall status
    if (health_data["environment"]["openai_api_key"] == "missing" or 
        health_data["environment"]["playwright"]["status"] == "missing" or
        health_data["environment"]["browser_use"]["status"] == "missing"):
        health_data["status"] = "unhealthy"
        
    return jsonify(health_data), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)