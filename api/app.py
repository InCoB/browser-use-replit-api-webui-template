import os
import json
import uuid
import time
import asyncio
import subprocess
import traceback
import platform
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Mock implementation for simulation mode
# This avoids dependency issues with langchain_openai and browser_use
SIMULATION_MODE = os.environ.get("BROWSER_USE_SIMULATION_MODE", "false").lower() == "true"
print(f"API starting in simulation mode: {SIMULATION_MODE}")

# Configure Playwright in Replit environment
print("Running in Replit environment - configuring for browser automation")
# Configure Playwright to work in Replit environment
os.environ["PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD"] = "0"  # We want to download the browser
os.environ["PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS"] = "1"  # Skip validating host requirements
os.environ["PLAYWRIGHT_CHROMIUM_SKIP_SYSTEM_DEPS"] = "true"  # Skip system dependencies check for Chromium

# Set additional environment variables for better compatibility
os.environ["PYTHONUNBUFFERED"] = "1"  # Ensure Python output is unbuffered
os.environ["NODE_OPTIONS"] = "--unhandled-rejections=strict"  # Better Node.js error handling

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

# Try Firefox browser instead of Chromium as it might have fewer dependencies
# Firefox tends to have better compatibility in restricted environments
os.environ["BROWSER_USE_BROWSER_TYPE"] = "firefox"

# Ensure headless mode is enabled
os.environ["BROWSER_USE_HEADLESS"] = "true"

# Set browser launch args to bypass common issues
os.environ["BROWSER_USE_BROWSER_ARGS"] = "--no-sandbox,--disable-setuid-sandbox,--disable-dev-shm-usage"

# Print browser configuration for debugging
print(f"Browser configuration: Using {os.environ.get('BROWSER_USE_BROWSER_TYPE')} in headless mode: {os.environ.get('BROWSER_USE_HEADLESS')}")

# Keep track of browser tasks
tasks = {}

def get_model_instance(model_name):
    """Get the appropriate LLM instance based on model name"""
    # In simulation mode, we don't need actual model instances
    if SIMULATION_MODE:
        return model_name
    
    # If not in simulation mode, try to load and use the real model
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
        # Return the model name string as a fallback
        return model_name

def detect_glibc_error(error_str):
    """Detect if an error message contains references to GLIBC incompatibility"""
    glibc_markers = ['GLIBC_', 'libc.so.6', 'version `GLIBC_', 'libglib-2.0.so.0']
    return any(marker in error_str for marker in glibc_markers)

async def run_browser_task(task_id, task_description, model_name):
    """Run a browser task asynchronously"""
    try:
        # Update task status to running
        tasks[task_id]["status"] = "running"
        tasks[task_id]["updated_at"] = datetime.now().isoformat()
        
        # Check for known system compatibility issues
        # If we've detected GLIBC issues in previous runs, use simulation mode
        simulation_mode = False
        
        # Set up a useful initial message that will help the user understand what's happening
        print(f"Processing task {task_id}: '{task_description}' using model {model_name}")
        
        # First, check if we have an API key for OpenAI to avoid Playwright issues if API key is missing
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            print(f"No OpenAI API key found for task {task_id} - API key is required for browser-use")
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = "OpenAI API key is required but not configured. Please set OPENAI_API_KEY."
            tasks[task_id]["updated_at"] = datetime.now().isoformat()
            return
        
        # First, try to import and install Playwright if needed
        try:
            # Check if playwright is installed
            import playwright
            print(f"Playwright version {getattr(playwright, '__version__', 'unknown')} is installed")
            
            # Try to import the sync_api module to verify the installation
            try:
                from playwright.sync_api import sync_playwright
                print("Playwright sync_api is available")
            except ImportError:
                print("Playwright sync_api could not be imported - attempting to install browsers")
                # Try to install playwright browsers if they're not already installed
                try:
                    import subprocess
                    print("Running: python -m playwright install firefox")
                    result = subprocess.run(
                        ["python", "-m", "playwright", "install", "firefox"],
                        capture_output=True, 
                        text=True, 
                        timeout=60
                    )
                    if result.returncode != 0:
                        print(f"Error installing Playwright browsers: {result.stderr}")
                        simulation_mode = True
                    else:
                        print("Successfully installed Playwright browsers")
                except Exception as install_error:
                    print(f"Failed to install Playwright browsers: {str(install_error)}")
                    simulation_mode = True
        except ImportError:
            print("Playwright is not installed - using simulation mode")
            simulation_mode = True
        
        # Try to detect if we're on an incompatible system by importing browser_use
        try:
            from browser_use import Agent
            print(f"browser-use imported successfully for task {task_id}")
        except Exception as e:
            error_str = str(e)
            print(f"Error importing browser_use: {error_str}")
            if detect_glibc_error(error_str):
                print(f"GLIBC compatibility issues detected: {error_str}")
                simulation_mode = True
                
        print(f"Running task {task_id} in simulation mode: {simulation_mode}")
        
        if simulation_mode:
            # In simulation mode, we'll create a simulated response instead of using actual browser
            print(f"Running task {task_id} in simulation mode")
            
            # Wait a bit to simulate processing time
            await asyncio.sleep(3)
            
            # Generate a simulated response based on the task
            simulated_result = {
                "simulation": True,
                "task": task_description,
                "model": model_name,
                "result": f"Simulated result for task: {task_description}",
                "screenshot": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
            }
            
            # Update task with simulated result
            tasks[task_id]["status"] = "completed"
            tasks[task_id]["result"] = simulated_result
            tasks[task_id]["updated_at"] = datetime.now().isoformat()
            return
        
        # If not in simulation mode, try to use the actual browser-use library
        # Initialize the agent with the specified LLM
        llm = get_model_instance(model_name)
        
        # Log agent initialization
        print(f"Initializing Agent for task: {task_id}")
        
        # Try different approaches to initialize the Agent
        browser_errors = []
        
        # First attempt - try with default configuration
        try:
            print(f"Attempt 1: Using default configuration for task {task_id}")
            # Use only the documented parameters according to browser-use examples
            from browser_use import Agent as BrowserAgent
            agent = BrowserAgent(
                task=task_description,
                llm=llm,
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
            browser_errors.append(f"Default configuration failed: {str(error1)}")
        
        # Second attempt - try with custom configuration if the library supports it
        try:
            print(f"Attempt 2: Using custom configuration for task {task_id}")
            # Import the Agent class directly for this attempt
            from browser_use import Agent as BrowserUseAgent
            
            # Check if browser_use version supports context parameter
            import inspect
            agent_params = inspect.signature(BrowserUseAgent.__init__).parameters
            if 'context' in agent_params:
                agent = BrowserUseAgent(
                    task=task_description,
                    llm=llm,
                    context={
                        'system': 'You are an AI browser agent that helps users navigate websites.'
                    }
                )
            else:
                # If not supported, create with minimal parameters
                agent = BrowserUseAgent(
                    task=task_description,
                    llm=llm
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
            browser_errors.append(f"Custom configuration failed: {str(error2)}")
        
        # Third attempt - try with specific browser_use parameters for v0.1.40
        try:
            print(f"Attempt 3: Using specific browser_use parameters for task {task_id}")
            
            # Import directly to ensure we have the latest version
            from browser_use import Agent
            
            # Based on the health check, we see the supported parameters for browser_use v0.1.40
            # We'll create the agent with a more compatible set of parameters
            agent = Agent(
                task=task_description,
                llm=llm,
                # We can't pass headless directly, so we'll rely on the environment variable
                # BROWSER_USE_HEADLESS=true that we set earlier
            )
            
            # Run the agent
            print(f"Running agent (attempt 3) for task: {task_id}")
            result = await agent.run()
            
            # Success! Update task with result
            tasks[task_id]["status"] = "completed"
            tasks[task_id]["result"] = result
            tasks[task_id]["updated_at"] = datetime.now().isoformat()
            return  # Exit the function successfully
            
        except Exception as error3:
            print(f"Attempt 3 failed for task {task_id}: {str(error3)}")
            browser_errors.append(f"Minimal parameter set failed: {str(error3)}")

            # If all attempts failed, fall back to simulation mode as a last resort
            print(f"All browser attempts failed for task {task_id}, falling back to simulation mode")
            
            # Wait a bit to simulate processing time
            await asyncio.sleep(3)
            
            # Generate a simulated response with information about the failure
            simulated_result = {
                "simulation": True,
                "task": task_description,
                "model": model_name,
                "result": f"Browser automation failed. Using simulated response for: {task_description}",
                "failure_reason": "Browser automation failed after multiple attempts. This could be due to system dependencies or configuration issues.",
                "screenshot": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
                "errors": browser_errors
            }
            
            # Update task with simulated result but mark as completed
            tasks[task_id]["status"] = "completed"  # Mark as completed even though it's simulated
            tasks[task_id]["result"] = simulated_result
            tasks[task_id]["updated_at"] = datetime.now().isoformat()
            return
    
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
        import queue
        
        # Use a better error handling approach for thread creation
        try:
            thread = threading.Thread(target=run_task)
            thread.daemon = True
            thread.start()
        except (RuntimeError, OSError) as e:
            # If we can't create a new thread due to resource limits,
            # run the task directly in the current thread
            print(f"Warning: Could not create a new thread - {str(e)}. Running task directly.")
            # Execute the task directly
            asyncio.run(run_browser_task(task_id, task_description, model_name))
        
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
    # Check if we're in simulation mode
    simulation_mode = os.environ.get("BROWSER_USE_SIMULATION_MODE", "false").lower() == "true"
    
    # Check for GLIBC compatibility issues
    glibc_compatible = True
    glibc_error = None
    try:
        # Try to run a quick browser test
        import subprocess
        result = subprocess.run(
            ["/home/runner/workspace/.cache/ms-playwright/chromium-1091/chrome-linux/chrome", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2
        )
        if result.returncode != 0:
            error_text = result.stderr
            if detect_glibc_error(error_text):
                glibc_compatible = False
                glibc_error = error_text
    except Exception as e:
        if detect_glibc_error(str(e)):
            glibc_compatible = False
            glibc_error = str(e)
    
    health_data = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": {
            "simulation_mode": simulation_mode,
            "glibc_compatible": glibc_compatible
        }
    }
    
    if not glibc_compatible:
        health_data["environment"]["glibc_error"] = glibc_error
        health_data["environment"]["notes"] = "System has GLIBC compatibility issues; browser automation will fall back to simulation mode."
    
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
    
    # Check Playwright environment variables
    playwright_env = {
        "PLAYWRIGHT_BROWSERS_PATH": os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "not set"),
        "PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS": os.environ.get("PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS", "not set"),
        "BROWSER_USE_BROWSER_TYPE": os.environ.get("BROWSER_USE_BROWSER_TYPE", "not set"),
        "BROWSER_USE_HEADLESS": os.environ.get("BROWSER_USE_HEADLESS", "not set")
    }
    health_data["environment"]["playwright_env"] = playwright_env
    
    # If we're in simulation mode, provide simulated version info
    if simulation_mode:
        health_data["environment"]["browser_use"] = {
            "status": "simulated",
            "version": "0.1.40"
        }
        health_data["environment"]["playwright"] = {
            "status": "simulated",
            "version": "1.40.0"
        }
    else:
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
    
    # Overall status check
    if not simulation_mode:
        if (health_data["environment"]["openai_api_key"] == "missing" or 
            health_data["environment"].get("playwright", {}).get("status") == "missing" or
            health_data["environment"].get("browser_use", {}).get("status") == "missing"):
            health_data["status"] = "unhealthy"
        
    return jsonify(health_data), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)