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

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Print the expected API key for debugging
print(f"API Key expected: {os.environ.get('EXTERNAL_API_KEY')}")

# Configure Playwright in Replit environment
print("Running in Replit environment - configuring for browser automation")

# Add ulimit settings to increase file descriptor and process limits
try:
    import resource
    # Try to increase the file descriptor limit
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f"Current file descriptor limits: soft={soft}, hard={hard}")
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    
    # Try to increase the process limit
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    print(f"Current process limits: soft={soft}, hard={hard}")
    resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
    
    print("Resource limits increased to maximum allowed values")
except Exception as e:
    print(f"Could not increase resource limits: {str(e)}")

# Configure Playwright to work in Replit environment
# Keep skip validate requirements, as it's used in the patch
os.environ["PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS"] = "1"
# We don't need Playwright to download browsers if we use Nix one exclusively
os.environ["PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD"] = "1"
# Keep skip system deps check
os.environ["PLAYWRIGHT_CHROMIUM_SKIP_SYSTEM_DEPS"] = "true"

# Set additional environment variables for better compatibility
os.environ["PYTHONUNBUFFERED"] = "1"  # Ensure Python output is unbuffered
os.environ["NODE_OPTIONS"] = "--unhandled-rejections=strict --max-old-space-size=256"  # Better Node.js error handling with limited memory

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
    # Add paths for newly installed dependencies
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

# Print the library path for debugging
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")

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
print(f"Browser configuration: Using Patched NIX Chromium (headless forced)")

# Keep track of browser tasks
tasks = {}

def get_model_instance(model_name):
    """Get the appropriate LLM instance based on model name"""
    # Always try to load and use the real model
    try:
        # Check if API key is available
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            raise ValueError("OpenAI API key not configured. Please set OPENAI_API_KEY in the .env file.")
        
        # Import here to allow the rest of the API to work even if this import fails
        from langchain_openai import ChatOpenAI
        
        # Define supported models and their configurations
        models = {
            "gpt-4o": ChatOpenAI(model="gpt-4o"),
            "gpt-4-turbo": ChatOpenAI(model="gpt-4-turbo"),
            "gpt-4": ChatOpenAI(model="gpt-4"),
            # gpt-3.5-turbo removed as per requirements
        }
        
        return models.get(model_name, models["gpt-4o"])  # Default to gpt-4o if model not found
    except Exception as e:
        print(f"Error initializing LLM model: {str(e)}")
        # If LLM fails to load, maybe raise an error or return None?
        # For now, returning model_name might cause issues later.
        # Consider raising the exception or handling it more gracefully.
        raise ValueError(f"Failed to initialize LLM {model_name}: {e}")

async def run_browser_task(task_id, task_description, model_name):
    """Run a browser task asynchronously using patched Playwright"""
    original_launch = None

    try:
        # Update task status to running
        tasks[task_id]["status"] = "running"
        tasks[task_id]["updated_at"] = datetime.now().isoformat()
        
        # --- Start of browser task logic (no simulation check) ---
        print(f"Processing task {task_id}: '{task_description}' using model {model_name}")
        
        # Check API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            print(f"No OpenAI API key found for task {task_id}")
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
            print(f"Playwright version {getattr(playwright, '__version__', 'unknown')} and browser-use version {getattr(browser_use, '__version__', 'unknown')} found.")
        except ImportError as e:
            print(f"Import error for Playwright or browser-use: {e}")
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = f"Required library not found: {e}"
            tasks[task_id]["updated_at"] = datetime.now().isoformat()
            return
        
        llm = get_model_instance(model_name)
        print(f"Initializing Agent for task: {task_id}")
        browser_error = None
        
        try:
            # Apply the monkey patch
            print(f"Applying Playwright patch for task {task_id}")
            chromium_path = os.environ.get('PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH')
            if not chromium_path or not os.path.exists(chromium_path):
                raise RuntimeError(f"NIX Chromium executable not found or path not set: {chromium_path}")

                print(f"Using NIX Chromium at path: {chromium_path}")

            # Store original launch method if not already stored
            if 'original_launch' not in locals() or original_launch is None:
                    original_launch = BrowserType.launch
                    
                    # Define our patched launch method
                    def patched_launch(self, **kwargs):
                    print(f"Patched Playwright launch called, forcing executablePath={chromium_path} and headless=True")
                    # Force the executable path to NIX Chromium
                        kwargs['executablePath'] = chromium_path
                    # Force headless mode to True
                    kwargs['headless'] = True
                    # Ensure env is properly initialized and set skip validation
                        if 'env' not in kwargs or kwargs['env'] is None:
                            kwargs['env'] = {}
                        if isinstance(kwargs['env'], dict):
                            kwargs['env']['PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS'] = 'true'
                    # Call the original stored launch method
                    if original_launch:
                        return original_launch(self, **kwargs)
                    else:
                        raise RuntimeError("Original Playwright launch method not captured.")
                    
                    # Apply the patch
                    BrowserType.launch = patched_launch
            print("Playwright monkey patch applied successfully for task.")

            # Configure and run the agent
            # Check if BrowserConfig is available and Agent supports it
            has_browser_config_param = 'browser_config' in inspect.signature(BrowserAgent.__init__).parameters
            if has_browser_config_param:
                print("Using BrowserConfig to initialize Agent.")
                browser_config = BrowserConfig(headless=True) # Patch handles executable path
                agent = BrowserAgent(
                    task=task_description,
                    llm=llm,
                    browser_config=browser_config,
                )
            else:
                print("BrowserConfig parameter not found in Agent, using default config.")
                agent = BrowserAgent(
                    task=task_description,
                    llm=llm,
                )
            
            print(f"Running agent for task: {task_id}")
            result = await agent.run()
            
            # Success! Update task with result
            tasks[task_id]["status"] = "completed"
            tasks[task_id]["result"] = result
            tasks[task_id]["updated_at"] = datetime.now().isoformat()
            print(f"Task {task_id} completed successfully.")

        except Exception as error:
            # Capture the error from the browser attempt
            browser_error = str(error)
            print(f"Browser task {task_id} failed: {browser_error}")
            traceback.print_exc() # Print detailed traceback for debugging

            # Mark task as failed - removed fallback to simulation
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = f"Browser automation failed: {browser_error}"
            tasks[task_id]["result"] = None # Ensure result is None on failure
            tasks[task_id]["updated_at"] = datetime.now().isoformat()

        finally:
             # Optional: Restore original launch method if captured
             if original_launch:
                 try:
                     BrowserType.launch = original_launch
                     print(f"Restored original Playwright launch method for task {task_id}.")
                 except NameError: # If BrowserType wasn't imported due to earlier error
                     pass
                 except Exception as e:
                     print(f"Could not restore original launch method: {e}")
    
    except Exception as e:
        print(f"General error processing task {task_id}: {str(e)}")
        traceback.print_exc()
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
        
        # Run the task directly in the current process without creating new threads
        # This is to avoid the "failed to create new OS thread" error
        print(f"Starting task {task_id} directly in the current process")
        
        # Use asyncio.create_task to run the browser task asynchronously
        # This avoids creating a new OS thread while still allowing the request to return
        async def start_task_async():
            loop = asyncio.get_event_loop()
            # Schedule the task to run in the background
            loop.create_task(run_browser_task(task_id, task_description, model_name))
        
        # Run the async function to schedule the task
        try:
            asyncio.run(start_task_async())
        except Exception as e:
            print(f"Error scheduling task {task_id}: {str(e)}")
            # If asyncio.run fails, try to run directly as a last resort
            # This might block the request, but it's better than failing completely
            try:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(lambda: asyncio.run(run_browser_task(task_id, task_description, model_name)))
            except Exception as e2:
                print(f"Could not run task {task_id} in any way: {str(e2)}")
                # Update the task status to reflect the failure
                tasks[task_id]["status"] = "failed"
                tasks[task_id]["error"] = f"Could not start task due to resource limits: {str(e2)}"
                tasks[task_id]["updated_at"] = datetime.now().isoformat()
        
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
@require_api_key
def get_browser_task(task_id):
    """Get the status and result of a browser task (requires API key)"""
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    
    # Create a JSON-serializable copy of the task data
    # Specifically handle the 'result' field which might contain non-serializable objects
    serializable_task = {}
    for key, value in task.items():
        if key == 'result':
            # Try to extract a simple text result if possible
            # Adjust this based on the actual structure of browser-use results
            if isinstance(value, dict) and 'text' in value:
                serializable_task[key] = value['text'] 
            elif isinstance(value, str):
                 serializable_task[key] = value
            elif value is not None:
                # Fallback: Convert potentially complex object to string representation
                # This might not be ideal JSON, but prevents the TypeError
                serializable_task[key] = str(value) 
            else:
                 serializable_task[key] = None
        else:
            # Assume other fields (id, status, timestamps, etc.) are serializable
            serializable_task[key] = value

    return jsonify(serializable_task), 200

@app.route("/api/supported-models", methods=["GET"])
def get_supported_models():
    """Get list of supported LLM models"""
    models = [
        {"id": "gpt-4o", "name": "GPT-4o"},
        {"id": "gpt-4-turbo", "name": "GPT-4 Turbo"},
        {"id": "gpt-4", "name": "GPT-4"},
        # gpt-3.5-turbo removed as per requirements
    ]
    return jsonify(models), 200

def get_system_resources():
    """Get available system resources"""
    try:
        import os
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
        return {"error": f"Could not collect system resources: {str(e)}"}

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    nix_chromium_working = False
    nix_chromium_output = "Test not run or failed."
    nix_chromium_path = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH")
    original_launch = None # Local scope for health check patch

    # Test NIX Chromium using the reliable patch
    if nix_chromium_path and os.path.exists(nix_chromium_path):
        try:
            print(f"Health Check: Testing NIX Chromium at {nix_chromium_path}")
            # Apply the same robust patch used in tasks
                    from playwright.sync_api import sync_playwright
                    from playwright._impl._browser_type import BrowserType
                    
            # Store original launch method
                    original_launch = BrowserType.launch
                    
            # Define our patched launch method for health check
            def health_patched_launch(self, **kwargs):
                print(f"Health Check: Patched launch forcing executablePath={nix_chromium_path} and headless=True")
                kwargs['executablePath'] = nix_chromium_path
                kwargs['headless'] = True
                        if 'env' not in kwargs or kwargs['env'] is None:
                            kwargs['env'] = {}
                        if isinstance(kwargs['env'], dict):
                            kwargs['env']['PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS'] = 'true'
                if original_launch:
                        return original_launch(self, **kwargs)
                else:
                    raise RuntimeError("Health Check: Original launch method not captured.")
                    
                    # Apply the patch
            BrowserType.launch = health_patched_launch
            print("Health check: Playwright monkey patch applied.")
                    
                    # Run a quick test with monkey-patched Playwright
                    with sync_playwright() as p:
                browser = p.chromium.launch() # Patch handles args
                version = browser.version
                        page = browser.new_page()
                        page.goto("http://example.com")
                        title = page.title()
                        page.close()
                        browser.close()
                        nix_chromium_working = True
                nix_chromium_output = f"Version: {version}. Successfully loaded '{title}' page."
                print("Health Check: NIX Chromium test successful.")
                    
        except Exception as e:
            nix_chromium_working = False
            nix_chromium_output = f"NIX Chromium test failed: {str(e)}"
            print(f"Health Check: NIX Chromium test failed: {e}")
            # traceback.print_exc() # Optional: print traceback for health check failures

        finally:
            # Restore the original method after health check test
             if original_launch:
                 try:
                     BrowserType.launch = original_launch
                     print("Health Check: Restored original Playwright launch method.")
                 except Exception as e:
                     print(f"Health Check: Failed to restore original launch method: {e}")

    else:
        nix_chromium_output = f"NIX Chromium path not set or invalid: {nix_chromium_path}"
        print(f"Health Check: {nix_chromium_output}")

    
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
        health_data["system_resources"] = {"error": "psutil not installed"}
    except Exception as e:
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
            
            # Try to check browsers installation
            try:
                from playwright.sync_api import sync_playwright
                with sync_playwright() as p:
                    health_data["environment"]["playwright"]["browsers"] = {
                        "chromium": "available" if hasattr(p, "chromium") else "missing",
                        "firefox": "available" if hasattr(p, "firefox") else "missing",
                        "webkit": "available" if hasattr(p, "webkit") else "missing"
                    }
            except Exception as browser_error:
                health_data["environment"]["playwright"]["browsers_error"] = str(browser_error)
                
        except ImportError as import_error:
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
            
            # Check browser_use Agent constructor parameters
            try:
                import inspect
                agent_params = inspect.signature(browser_use.Agent.__init__).parameters
                param_names = list(agent_params.keys())
                # Remove 'self' from the list if present
                if 'self' in param_names:
                    param_names.remove('self')
                health_data["environment"]["browser_use"]["supported_params"] = param_names
            except Exception as param_error:
                health_data["environment"]["browser_use"]["param_error"] = str(param_error)
        except ImportError as import_error:
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
                system_deps[lib] = f"check failed: {str(e)}"
        
        health_data["environment"]["system_dependencies"] = system_deps
    
    # Update overall status check (remove simulation mode condition)
    api_key_ok = health_data["environment"]["openai_api_key"] == "available"
    playwright_ok = health_data["environment"].get("playwright", {}).get("status") == "installed"
    browser_use_ok = health_data["environment"].get("browser_use", {}).get("status") == "installed"

    if not all([api_key_ok, playwright_ok, browser_use_ok, nix_chromium_working]):
            health_data["status"] = "unhealthy"
        # Add more specific notes if needed
        if not api_key_ok: health_data["environment"]["notes"] = health_data["environment"].get("notes", "") + " OpenAI API Key missing."
        if not playwright_ok: health_data["environment"]["notes"] = health_data["environment"].get("notes", "") + " Playwright missing/error."
        if not browser_use_ok: health_data["environment"]["notes"] = health_data["environment"].get("notes", "") + " browser-use missing/error."
        # Note for nix chromium failure already added above
        
    return jsonify(health_data), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)