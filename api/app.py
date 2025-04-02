import os
import json
import uuid
import time
import asyncio
import subprocess
import traceback
import platform
import inspect
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from .auth import require_api_key
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from .health_check_logic import perform_detailed_diagnostics

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# --- Logging Configuration START ---
import logging
from logging.handlers import RotatingFileHandler

# Disable Flask's default logging handlers
app.logger.handlers.clear()
app.logger.propagate = False # Prevent messages from propagating to the root logger

# Determine log level from environment variable (default to INFO)
log_level_name = os.environ.get("FLASK_LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)

# Create formatter
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Configure file handler
file_handler = RotatingFileHandler('api.log', maxBytes=1024*1024*5, backupCount=2) # 5MB max size, 2 backups
file_handler.setFormatter(log_formatter)
file_handler.setLevel(log_level)

# Configure stream handler (console)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
stream_handler.setLevel(log_level)

# Add OUR handlers to the Flask app's logger
app.logger.addHandler(file_handler)
app.logger.addHandler(stream_handler)
app.logger.setLevel(log_level)

# Also configure the root logger if needed (e.g., for libraries)
# logging.basicConfig(level=log_level, handlers=[file_handler, stream_handler])

# Silence Werkzeug's default logger if we are managing logging
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.handlers.clear() # Remove default handler
werkzeug_logger.propagate = False
werkzeug_logger.setLevel(logging.WARNING) # Or set level as needed
# --- Logging Configuration END ---

# Print startup message using the logger
app.logger.info("Starting Browser Companion API...")
app.logger.info(f"Environment: {'Replit' if os.environ.get('REPL_ID') else 'Local'}")
app.logger.info(f"Log Level: {log_level_name}")

# Configure Playwright in Replit environment
if os.environ.get('REPL_ID'):
    app.logger.debug("Replit environment detected, configuring resource limits and Playwright...")
# Add ulimit settings to increase file descriptor and process limits
try:
    import resource
    # Try to increase the file descriptor limit
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        app.logger.debug(f"File descriptor limit set to {hard}")
    
    # Try to increase the process limit
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
        app.logger.debug(f"Process limit set to {hard}")
except Exception as e:
        app.logger.warning(f"Could not increase resource limits: {str(e)}")

    # Configure Playwright environment variables
    os.environ["PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS"] = "1"
    os.environ["PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD"] = "1"
    os.environ["PLAYWRIGHT_CHROMIUM_SKIP_SYSTEM_DEPS"] = "true"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["NODE_OPTIONS"] = "--unhandled-rejections=strict --max-old-space-size=256"

    # Configure browser settings
    os.environ["BROWSER_USE_BROWSER_TYPE"] = "chromium"
    os.environ["BROWSER_USE_HEADLESS"] = "true"
    os.environ["BROWSER_USE_BROWSER_ARGS"] = "--no-sandbox,--disable-setuid-sandbox,--disable-dev-shm-usage,--disable-gpu,--disable-software-rasterizer,--disable-extensions,--single-process,--no-zygote"
    os.environ["BROWSER_USE_MAX_THREADS"] = "1"

    app.logger.debug("Playwright environment variables configured for Replit.")
    app.logger.debug(f"LD_LIBRARY_PATH set to: {os.environ.get('LD_LIBRARY_PATH')}")

# Print the expected API key for debugging (only in DEBUG mode)
app.logger.debug(f"Expected API Key: {os.environ.get('EXTERNAL_API_KEY')}")

# Set environment variables from .env if not already set
if "PLAYWRIGHT_BROWSERS_PATH" not in os.environ:
    browsers_path = os.getenv("PLAYWRIGHT_BROWSERS_PATH")
    if browsers_path:
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = browsers_path

# Use chromium since we now have the updated GLIBC
os.environ["BROWSER_USE_BROWSER_TYPE"] = "chromium"

# Ensure headless mode is enabled
os.environ["BROWSER_USE_HEADLESS"] = "true"

# Set browser launch args to bypass common issues and use fewer resources
os.environ["BROWSER_USE_BROWSER_ARGS"] = "--no-sandbox,--disable-setuid-sandbox,--disable-dev-shm-usage,--disable-gpu,--disable-software-rasterizer,--disable-extensions,--single-process,--no-zygote"

# Set resource limits to avoid thread creation issues
os.environ["BROWSER_USE_MAX_THREADS"] = "1"  # Limit thread usage

# Print browser configuration for debugging
app.logger.info(f"Browser configuration: Using Patched NIX Chromium (headless forced)")

# Keep track of browser tasks
tasks = {}

def get_model_instance(model_name):
    """Get the appropriate LLM instance based on model name"""
    try:
        if model_name.startswith("gemini-"):
            # Handle Google Gemini Models
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                app.logger.error("Google API key not configured. Please set GOOGLE_API_KEY.")
                raise ValueError("Google API key not configured.")
            app.logger.info(f"Initializing Google Gemini model: {model_name}")
            llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
            return llm
        elif model_name.startswith("gpt-"):
            # Handle OpenAI Models
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key or api_key == "your_openai_api_key_here":
                app.logger.error("OpenAI API key not configured. Please set OPENAI_API_KEY.")
                raise ValueError("OpenAI API key not configured.")
            app.logger.info(f"Initializing OpenAI model: {model_name}")
            llm = ChatOpenAI(model=model_name, openai_api_key=api_key)
            return llm
        else:
            # Fallback or default model (e.g., default to gpt-4o)
            app.logger.warning(f"Unknown model prefix for '{model_name}'. Defaulting to gpt-4o.")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
                app.logger.error("OpenAI API key not configured for default model.")
                raise ValueError("OpenAI API key not configured for default model.")
            llm = ChatOpenAI(model="gpt-4o", openai_api_key=api_key)
            return llm

    except Exception as e:
        app.logger.error(f"Error initializing LLM model '{model_name}': {str(e)}")
        raise ValueError(f"Failed to initialize LLM {model_name}: {e}")

async def run_browser_task(task_id, task_description, model_name):
    """Run a browser task asynchronously using patched Playwright"""
    original_launch = None
    app.logger.info(f"Starting browser task {task_id}...")

    try:
        # Update task status to running
        tasks[task_id]["status"] = "running"
        tasks[task_id]["updated_at"] = datetime.now().isoformat()
        app.logger.info(f"Task {task_id} status set to running.")
        
        # --- Start of browser task logic (no simulation check) ---
        app.logger.debug(f"Processing task {task_id}: '{task_description}' using model {model_name}")
        
        # Check API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            app.logger.error(f"No OpenAI API key found for task {task_id}")
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = "OpenAI API key required but not configured."
            tasks[task_id]["updated_at"] = datetime.now().isoformat()
            return
        
        # Check essential imports
        try:
            import playwright
            from playwright._impl._browser_type import BrowserType
            import browser_use
            from browser_use import Agent as BrowserAgent, BrowserConfig
            app.logger.debug(f"Playwright version {getattr(playwright, '__version__', 'unknown')} and browser-use version {getattr(browser_use, '__version__', 'unknown')} found.")
        except ImportError as e:
            app.logger.error(f"Import error for Playwright or browser-use: {e}")
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = f"Required library not found: {e}"
            tasks[task_id]["updated_at"] = datetime.now().isoformat()
            return
        
        llm = get_model_instance(model_name)
        app.logger.debug(f"Initializing Agent for task: {task_id}")
        browser_error = None
        
        try:
            # Apply the monkey patch
            app.logger.debug(f"Applying Playwright patch for task {task_id}")
            chromium_path = os.environ.get('PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH')
            if not chromium_path or not os.path.exists(chromium_path):
                app.logger.error(f"NIX Chromium executable not found or path not set: {chromium_path}")
                raise RuntimeError(f"NIX Chromium executable not found or path not set: {chromium_path}")

            app.logger.debug(f"Using NIX Chromium at path: {chromium_path}")

            # Store original launch method if not already stored
            if 'original_launch' not in locals() or original_launch is None:
                    original_launch = BrowserType.launch
                app.logger.debug("Original Playwright launch method captured.")
                    
                    # Define our patched launch method
                    def patched_launch(self, **kwargs):
                    app.logger.debug(f"Patched Playwright launch called, forcing executablePath={chromium_path} and headless=True")
                        kwargs['executablePath'] = chromium_path
                    kwargs['headless'] = True
                        if 'env' not in kwargs or kwargs['env'] is None:
                            kwargs['env'] = {}
                        if isinstance(kwargs['env'], dict):
                            kwargs['env']['PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS'] = 'true'
                    if original_launch:
                        return original_launch(self, **kwargs)
                    else:
                        app.logger.error("Original Playwright launch method not captured before patching.")
                        raise RuntimeError("Original Playwright launch method not captured.")
                    
                    # Apply the patch
                    BrowserType.launch = patched_launch
                app.logger.debug("Playwright monkey patch applied.")
            else:
                app.logger.debug("Playwright patch already applied.")

            # Configure and run the agent
            has_browser_config_param = 'browser_config' in inspect.signature(BrowserAgent.__init__).parameters
            if has_browser_config_param:
                app.logger.debug("Using BrowserConfig to initialize Agent.")
                browser_config = BrowserConfig(headless=True)
                agent = BrowserAgent(
                    task=task_description,
                    llm=llm,
                    browser_config=browser_config,
                )
            else:
                app.logger.debug("BrowserConfig parameter not found in Agent, using default config.")
                agent = BrowserAgent(
                    task=task_description,
                    llm=llm,
                )
            
            app.logger.info(f"Running agent for task: {task_id}")
            result = await agent.run()
            
            # Success! Update task with result
            tasks[task_id]["status"] = "completed"
            tasks[task_id]["result"] = result
            tasks[task_id]["updated_at"] = datetime.now().isoformat()
            app.logger.info(f"Task {task_id} completed successfully.")

        except Exception as error:
            # Capture the error from the browser attempt
            browser_error = str(error)
            app.logger.error(f"Browser task {task_id} failed: {browser_error}", exc_info=True) # Log with traceback

            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = f"Browser automation failed: {browser_error}"
            tasks[task_id]["result"] = None
            tasks[task_id]["updated_at"] = datetime.now().isoformat()

        finally:
             # Optional: Restore original launch method if captured
             if original_launch:
                 try:
                     BrowserType.launch = original_launch
                     app.logger.debug(f"Restored original Playwright launch method for task {task_id}.")
                 except NameError:
                     pass
                 except Exception as e:
                     app.logger.warning(f"Could not restore original launch method: {e}")
    
    except Exception as e:
        app.logger.error(f"General error processing task {task_id}: {str(e)}", exc_info=True)
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = f"Unexpected error: {str(e)}"
        tasks[task_id]["updated_at"] = datetime.now().isoformat()

@app.route("/api/browser-tasks", methods=["POST"])
@require_api_key
def create_browser_task():
    try:
        data = request.json
        if not data or "task" not in data:
            return jsonify({"error": "Missing 'task' in request body"}), 400

        task_description = data["task"]
        model_name = data.get("model", "gpt-4o") # Default to gpt-4o
        task_id = str(uuid.uuid4())
        
        tasks[task_id] = {
            "id": task_id,
            "task": task_description,
            "model": model_name,
            "status": "pending",
            "result": None,
            "error": None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        app.logger.info(f"Received task creation request: ID {task_id}, Model {model_name}")

        # --- Restore ThreadPoolExecutor logic for scheduling --- 
        app.logger.debug(f"Attempting to schedule task {task_id}...")
        try:
            # Use ThreadPoolExecutor to run the async task in a separate thread
            import concurrent.futures
            
            # Define the function to be run in the thread pool
            def run_async_in_thread():
                app.logger.debug(f"Background thread started for task {task_id}.")
                # Create and manage a dedicated event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                app.logger.debug(f"New event loop created for task {task_id} in thread.")
                try:
                    # Run the main async task function until it completes
                    loop.run_until_complete(run_browser_task(task_id, task_description, model_name))
                    app.logger.debug(f"run_browser_task {task_id} completed in background thread loop.")
                except Exception as thread_e:
                    # Log errors happening within the async task execution itself
                    app.logger.error(f"Error executing run_browser_task {task_id} in background thread: {thread_e}", exc_info=True)
                    # Update task status from within the thread if possible (might be tricky if task dict access is not thread-safe)
                    # Safest might be to rely on polling finding the error later or add thread-safe queue
                finally:
                    # Ensure the loop is closed
                    app.logger.debug(f"Closing event loop for task {task_id}.")
                    loop.close()
                    app.logger.debug(f"Event loop closed for task {task_id}.")

            # Submit the wrapper function to the thread pool executor
            # Using only 1 worker to prevent excessive resource use per request
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_async_in_thread)
                app.logger.debug(f"Task {task_id} submitted via ThreadPoolExecutor using dedicated loop.")
                # Optional: could add future.add_done_callback() here 
                # to log if the submit itself raised an exception

        except Exception as e:
            # Catch errors during the scheduling/submission itself
            app.logger.error(f"Failed to schedule task {task_id} using ThreadPoolExecutor: {e}", exc_info=True)
                tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = f"Failed to start task: {e}"
                tasks[task_id]["updated_at"] = datetime.now().isoformat()
            # Return error immediately if scheduling fails
            return jsonify({"error": f"Failed to schedule task: {e}"}), 500
        # --- End Restore --- 
             
        # REMOVED: Simplified asyncio.get_event_loop / asyncio.run logic
        
        # Return 202 Accepted immediately after scheduling attempt
        return jsonify({"id": task_id, "status": "pending"}), 202 
    
    except Exception as e:
        # Catch any other unexpected errors during request handling
        app.logger.error(f"Error in create_browser_task endpoint processing request: {e}", exc_info=True)
        return jsonify({"error": "Internal server error during task creation processing."}), 500

@app.route("/api/browser-tasks/<task_id>", methods=["GET"])
@require_api_key
def get_browser_task(task_id):
    """Get details of a specific browser task"""
    app.logger.debug(f"Request received for task details: {task_id}")
    if task_id not in tasks:
        app.logger.warning(f"Task not found: {task_id}")
        return jsonify({"error": "Task not found"}), 404
    
    task = tasks[task_id]
    # Log task status only if it has changed since last logged (or if never logged)
    last_logged = task.get("_last_logged_status")
    current_status = task["status"]
    if last_logged != current_status:
        app.logger.info(f"Task {task_id} status: {current_status}")
        # Update the internal tracking variable
        tasks[task_id]["_last_logged_status"] = current_status 
        
    # Create a JSON-serializable copy of the task data
    # Specifically handle the 'result' field which might contain non-serializable objects
    response_task = {}
    for key, value in task.items():
        if key == 'result':
            # Try to extract a simple text result if possible (adjust as needed)
            if isinstance(value, dict) and 'text' in value:
                response_task[key] = value['text'] 
            elif isinstance(value, str):
                 response_task[key] = value
            elif value is not None:
                # Fallback: Convert complex object to string representation
                app.logger.debug(f"Converting non-serializable result for task {task_id} to string.")
                response_task[key] = str(value) 
            else:
                 response_task[key] = None
        elif key != "_last_logged_status": # Exclude internal tracking field
            # Assume other fields are serializable
            response_task[key] = value

    # Add detailed debugging right before jsonify
    app.logger.debug(f"[DEBUG] Type of response_task['result'] before jsonify: {type(response_task.get('result'))}")
    try:
        app.logger.debug(f"[DEBUG] Content of response_task before jsonify: {json.dumps(response_task, indent=2, default=str)}") # Try to dump with str fallback
    except Exception as dump_error:
        app.logger.debug(f"[DEBUG] Could not dump response_task, content: {response_task}")

    return jsonify(response_task)

@app.route("/api/supported-models", methods=["GET"])
def get_supported_models():
    """Get list of supported LLM models"""
    app.logger.debug("Fetching list of supported models.")
    models = [
        # OpenAI Models
        {"id": "gpt-4o", "name": "GPT-4o"},
       

        # Google Models
        {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash"},
        {"id": "gemini-2.0-flash-lite", "name": "Gemini 2.0 Flash-Lite"},
        
    ]
    models.sort(key=lambda x: x['name'])
    return jsonify(models), 200

# Simplified Health Check Endpoint
@app.route("/health", methods=["GET"])
def health_check():
    """Basic health check endpoint. Returns simple status."""
    app.logger.debug("Basic health check endpoint called.")
    return jsonify({"status": "healthy"}), 200

# New Detailed Diagnostics Endpoint
@app.route("/diagnostics", methods=["GET"])
@require_api_key # Protect this endpoint
def diagnostics():
    """Runs detailed environment diagnostics and returns the results."""
    app.logger.info("Detailed diagnostics endpoint called...")
    # Call the function from the other file, passing the app's logger
    diagnostic_data = perform_detailed_diagnostics(app.logger)
    # Determine status code based on results
    status_code = 200 if diagnostic_data.get("overall_status") == "healthy" else 503 # Service Unavailable
    return jsonify(diagnostic_data), status_code

if __name__ == "__main__":
    # Use Flask's default run for development, which handles reloading.
    # Logging is configured above using app.logger.
    app.run(host="0.0.0.0", port=5001, debug=os.environ.get("FLASK_ENV") == "development")