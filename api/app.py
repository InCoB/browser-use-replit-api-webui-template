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

    # Add paths to the system libraries
    library_paths = [
        "/nix/store/5gn1p7qbij0i7lbj9xdvpz1rrngxiydw-xorg-libxcb-1.17.0/lib",
        "/nix/store/3jfj4q2w92kkd2dff9p9bj22b0f1ibdc-libxkbcommon-1.6.0/lib",
        "/nix/store/byjgb9md2gk4a9hfsflnlmn7g6wddmdc-libdrm-2.4.120/lib",
        "/nix/store/v0jmnz5ssrxlm4l6ci80yaqgpi3wc87z-libXrandr-1.5.4/lib",
        "/nix/store/k65rj40r9kg7vci0c9irhg8b2n4frga8-libXcomposite-0.4.6/lib",
        "/nix/store/hhxfpbs54w1mmgzsncbs4yh9gvld6yls-libXdamage-1.1.6/lib",
        "/nix/store/4xm83ky7gvq9h5gzl5ylj1s3b1r38wj9-libXfixes-6.0.1/lib",
        "/nix/store/rvcg06hzzxzq2ks2ag9jffjlc1m3zc09-libXrender-0.9.11/lib",
        "/nix/store/jgxvbid3i7z7qiw55r5qwpgcfpchvkhb-libXtst-1.2.4/lib",
        "/nix/store/d1jplxanpq0g2k9s0jgrks11a9hlj2r2-libXi-1.8.1/lib",
        "/nix/store/wkdvprbvp7gg2qcvr1mlj3ndlk2p9b9b-pango-1.50.14/lib",
        "/nix/store/yfrfsxp44ln4lywhzlfj3gkndmdvhl52-glib-2.78.3/lib",
        "/nix/store/irkif55f313pzgs5n6dpqxx9hk2q5y57-nss-3.95/lib",
        "/nix/store/f1z5vnsw4r6yz13a7q2xi4sjps41pn6m-alsa-lib-1.2.10/lib",
        "/nix/store/6rrk0i1qxs5sq9jl1ycys3h6r3f4d8sl-at-spi2-atk-2.46.0/lib",
        "/nix/store/4wkwv40iqxnmkw25y618qdqmwcbpi4z3-cups-2.4.7/lib",
        "/nix/store/mpy304iwc2fk1n4n5x4cfmvss2xmvny0-dbus-1.14.10/lib"
    ]

    # Join all library paths and add them to LD_LIBRARY_PATH
    lib_path_str = ":".join(library_paths)
    if "LD_LIBRARY_PATH" in os.environ:
        os.environ["LD_LIBRARY_PATH"] = f"{lib_path_str}:{os.environ['LD_LIBRARY_PATH']}"
    else:
        os.environ["LD_LIBRARY_PATH"] = lib_path_str

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
    """Create a new browser task (requires API key)"""
    try:
        data = request.json
        task_description = data.get("task")
        model_name = data.get("model", "gpt-4o")
        
        if not task_description:
            app.logger.warning("Create task request missing task description.")
            return jsonify({"error": "Task description is required"}), 400
        
        task_id = str(uuid.uuid4())
        app.logger.info(f"Creating new task {task_id} for: '{task_description[:50]}...'")
        
        tasks[task_id] = {
            "id": task_id,
            "task": task_description,
            "model": model_name,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        
        app.logger.debug(f"Starting task {task_id} scheduling...")
        
        async def start_task_async():
            loop = asyncio.get_event_loop()
            loop.create_task(run_browser_task(task_id, task_description, model_name))
            app.logger.debug(f"Task {task_id} scheduled on event loop.")
        
        try:
            asyncio.run(start_task_async())
        except RuntimeError as e:
            if "cannot run nested event loops" in str(e).lower():
                app.logger.warning(f"Nested event loop detected for task {task_id}. Trying ThreadPoolExecutor.")
                try:
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        executor.submit(lambda: asyncio.run(run_browser_task(task_id, task_description, model_name)))
                        app.logger.debug(f"Task {task_id} submitted via ThreadPoolExecutor.")
                except Exception as e2:
                    app.logger.error(f"Could not run task {task_id} via ThreadPoolExecutor: {str(e2)}", exc_info=True)
                    tasks[task_id]["status"] = "failed"
                    tasks[task_id]["error"] = f"Could not start task due to resource limits: {str(e2)}"
                    tasks[task_id]["updated_at"] = datetime.now().isoformat()
            else:
                 app.logger.error(f"Error scheduling task {task_id} with asyncio.run: {str(e)}", exc_info=True)
                 tasks[task_id]["status"] = "failed"
                 tasks[task_id]["error"] = f"Could not start task due to asyncio error: {str(e)}"
                 tasks[task_id]["updated_at"] = datetime.now().isoformat()
        except Exception as e:
            app.logger.error(f"Unexpected error scheduling task {task_id}: {str(e)}", exc_info=True)
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = f"Could not start task due to unexpected error: {str(e)}"
            tasks[task_id]["updated_at"] = datetime.now().isoformat()
        
        return jsonify({
            "id": task_id,
            "status": tasks[task_id]["status"], # Return the potentially updated status
            "message": "Task creation initiated."
        }), 201
    
    except Exception as e:
        app.logger.error(f"Error in create_browser_task endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error during task creation."}), 500

@app.route("/api/browser-tasks", methods=["GET"])
def list_browser_tasks():
    """List all browser tasks"""
    app.logger.debug("Listing all browser tasks.")
    # Return a deep copy to avoid modifying the original task data when logging status
    tasks_list = [json.loads(json.dumps(task, default=str)) for task in tasks.values()] 
    return jsonify(tasks_list), 200

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

def get_system_resources():
    """Get available system resources"""
    try:
        import psutil
        
        # Get memory stats
        mem = psutil.virtual_memory()
        
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Get process info
        process = psutil.Process(os.getpid())
        process_mem = process.memory_info()
        
        # Get open file count
        try:
            open_files = len(process.open_files())
        except:
            open_files = -1
            
        # Get thread count
        try:
            thread_count = len(process.threads())
        except:
            thread_count = -1
            
        return {
            "memory": {
                "total": mem.total,
                "available": mem.available,
                "used": mem.used,
                "percent": mem.percent
            },
            "cpu_percent": cpu_percent,
            "process": {
                "memory_used": process_mem.rss,
                "memory_percent": process.memory_percent(),
                "open_files": open_files,
                "thread_count": thread_count
            }
        }
    except Exception as e:
        app.logger.warning(f"Could not collect system resources: {str(e)}")
        return {"error": f"Could not collect system resources: {str(e)}"}

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    app.logger.info("Performing health check...")
    nix_chromium_working = False
    nix_chromium_output = "Test not run or failed."
    nix_chromium_path = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH")
    original_launch = None

    if nix_chromium_path and os.path.exists(nix_chromium_path):
        try:
            app.logger.debug(f"Health Check: Testing NIX Chromium at {nix_chromium_path}")
            from playwright.sync_api import sync_playwright
            from playwright._impl._browser_type import BrowserType
            
            original_launch = BrowserType.launch
            
            def health_patched_launch(self, **kwargs):
                app.logger.debug(f"Health Check: Patched launch forcing executablePath={nix_chromium_path} and headless=True")
                kwargs['executablePath'] = nix_chromium_path
                kwargs['headless'] = True
                if 'env' not in kwargs or kwargs['env'] is None:
                    kwargs['env'] = {}
                if isinstance(kwargs['env'], dict):
                    kwargs['env']['PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS'] = 'true'
                if original_launch:
                    return original_launch(self, **kwargs)
                else:
                    app.logger.error("Health Check: Original launch method not captured.")
                    raise RuntimeError("Health Check: Original launch method not captured.")
            
            BrowserType.launch = health_patched_launch
            app.logger.debug("Health check: Playwright monkey patch applied.")
            
            with sync_playwright() as p:
                browser = p.chromium.launch() 
                version = browser.version
                page = browser.new_page()
                page.goto("http://example.com")
                title = page.title()
                page.close()
                browser.close()
                nix_chromium_working = True
                nix_chromium_output = f"Version: {version}. Successfully loaded '{title}' page."
                app.logger.info("Health Check: NIX Chromium test successful.")
                    
        except Exception as e:
            nix_chromium_working = False
            nix_chromium_output = f"NIX Chromium test failed: {str(e)}"
            app.logger.error(f"Health Check: NIX Chromium test failed: {e}", exc_info=True)

        finally:
            if original_launch:
                 try:
                     BrowserType.launch = original_launch
                     app.logger.debug("Health Check: Restored original Playwright launch method.")
                 except Exception as e:
                     app.logger.warning(f"Health Check: Failed to restore original launch method: {e}")

    else:
        nix_chromium_output = f"NIX Chromium path not set or invalid: {nix_chromium_path}"
        app.logger.warning(f"Health Check: {nix_chromium_output}")

    health_data = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": {
            "nix_chromium": {
                "working": nix_chromium_working,
                "path": nix_chromium_path if nix_chromium_path else "not set",
                "output": nix_chromium_output
            }
        }
    }
    
    # Update notes logic
    if not nix_chromium_working:
         health_data["status"] = "unhealthy"
         health_data["environment"]["notes"] = f"NIX Chromium test failed. Check logs. Output: {nix_chromium_output}"
    else: # Simplified from elif nix_chromium_working
         health_data["environment"]["notes"] = "NIX Chromium appears to be working correctly."
    
    # Add system resource information
    try:
        import psutil
        health_data["system_resources"] = get_system_resources()
    except ImportError:
        app.logger.warning("psutil not installed, cannot report system resources.")
        health_data["system_resources"] = {"error": "psutil not installed"}
    except Exception as e:
        app.logger.error(f"Error getting system resources: {str(e)}", exc_info=True)
        health_data["system_resources"] = {"error": f"Error getting system resources: {str(e)}"}
    
    # Check OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    health_data["environment"]["openai_api_key"] = "available" if api_key and api_key != "your_openai_api_key_here" else "missing"
    
    # Get Python version
    import sys
    health_data["environment"]["python"] = {
        "version": sys.version,
        "executable": sys.executable
    }
    
    # Check Playwright environment variables (update list)
    playwright_env = {
        "PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS": os.environ.get("PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS", "not set"),
        "PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH": os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH", "not set"),
        "BROWSER_USE_BROWSER_ARGS": os.environ.get("BROWSER_USE_BROWSER_ARGS", "not set") # Kept this one
    }
    
    # Check if NIX Chromium exists at the specified path
    chromium_path = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH")
    if chromium_path:
        playwright_env["NIX_CHROMIUM_EXISTS"] = "yes" if os.path.exists(chromium_path) else "no"
        # Check if the file is executable
        if os.path.exists(chromium_path):
            try:
                playwright_env["NIX_CHROMIUM_EXECUTABLE"] = "yes" if os.access(chromium_path, os.X_OK) else "no"
            except Exception as e:
                playwright_env["NIX_CHROMIUM_EXECUTABLE"] = f"check failed: {str(e)}"
    
    health_data["environment"]["playwright_env"] = playwright_env
    
    # Check installed packages and versions
    try:
        import playwright
        health_data["environment"]["playwright"] = {
            "status": "installed",
            "version": getattr(playwright, "__version__", "unknown")
        }
        app.logger.debug("Playwright is installed.")
        
        # Try to check browsers installation
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                health_data["environment"]["playwright"]["browsers"] = {
                    "chromium": "available" if hasattr(p, "chromium") else "missing",
                    "firefox": "available" if hasattr(p, "firefox") else "missing",
                    "webkit": "available" if hasattr(p, "webkit") else "missing"
                }
                app.logger.debug("Playwright browsers checked.")
        except Exception as browser_error:
            app.logger.warning(f"Error checking Playwright browsers: {browser_error}")
            health_data["environment"]["playwright"]["browsers_error"] = str(browser_error)
            
    except ImportError as import_error:
        app.logger.warning(f"Playwright not installed: {import_error}")
        health_data["environment"]["playwright"] = {
            "status": "missing",
            "error": str(import_error)
        }
    
    try:
        import browser_use
        health_data["environment"]["browser_use"] = {
            "status": "installed",
            "version": getattr(browser_use, "__version__", "unknown")
        }
        app.logger.debug("browser-use is installed.")
        
        # Check browser_use Agent constructor parameters
        try:
            import inspect
            agent_params = inspect.signature(browser_use.Agent.__init__).parameters
            param_names = list(agent_params.keys())
            # Remove 'self' from the list if present
            if 'self' in param_names:
                param_names.remove('self')
            health_data["environment"]["browser_use"]["supported_params"] = param_names
            app.logger.debug(f"browser_use Agent params: {param_names}")
        except Exception as param_error:
            app.logger.warning(f"Error checking browser_use params: {param_error}")
            health_data["environment"]["browser_use"]["param_error"] = str(param_error)
    except ImportError as import_error:
        app.logger.warning(f"browser-use not installed: {import_error}")
        health_data["environment"]["browser_use"] = {
            "status": "missing",
            "error": str(import_error)
        }
    
    # Check for system libraries that Playwright needs
    system_deps = {}
    for lib in ["libnss3", "libxrandr2", "libgbm1", "libxshmfence1", "libdrm2"]:
        try:
            result = subprocess.run(["ldconfig", "-p"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if lib in result.stdout:
                system_deps[lib] = "found"
            else:
                system_deps[lib] = "not found in ldconfig"
        except Exception as e:
            app.logger.warning(f"System dependency check failed for {lib}: {e}")
            system_deps[lib] = f"check failed: {str(e)}"
    health_data["environment"]["system_dependencies"] = system_deps
    app.logger.debug(f"System dependencies check: {system_deps}")

    # Update overall status check (remove simulation mode condition)
    api_key_ok = health_data["environment"]["openai_api_key"] == "available"
    playwright_ok = health_data["environment"].get("playwright", {}).get("status") == "installed"
    browser_use_ok = health_data["environment"].get("browser_use", {}).get("status") == "installed"

    if not all([api_key_ok, playwright_ok, browser_use_ok, nix_chromium_working]):
        health_data["status"] = "unhealthy"
        # Add more specific notes if needed
        if not api_key_ok: 
            health_data["environment"]["notes"] = health_data["environment"].get("notes", "") + " OpenAI API Key missing."
        if not playwright_ok: 
            health_data["environment"]["notes"] = health_data["environment"].get("notes", "") + " Playwright missing/error."
        if not browser_use_ok: 
            health_data["environment"]["notes"] = health_data["environment"].get("notes", "") + " browser-use missing/error."
        # Note for nix chromium failure already added above
        
    app.logger.info(f"Health check completed. Status: {health_data['status']}")
    return jsonify(health_data), 200

if __name__ == "__main__":
    # Use Flask's default run for development, which handles reloading.
    # Logging is configured above using app.logger.
    app.run(host="0.0.0.0", port=5001, debug=os.environ.get("FLASK_ENV") == "development")