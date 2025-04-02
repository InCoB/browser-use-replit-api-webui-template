# api/health_check_logic.py
import os
import sys
import json
import subprocess
import platform
import inspect
from datetime import datetime
import logging # Needed for type hinting the logger argument

# --- System Resource Function ---
def get_system_resources(logger):
    """Get available system resources, using the provided logger."""
    logger.debug("Collecting system resource information...")
    try:
        import psutil
        mem = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        process = psutil.Process(os.getpid())
        process_mem = process.memory_info()
        open_files, thread_count = "N/A", "N/A" # Default values

        try: open_files = len(process.open_files())
        except psutil.AccessDenied: logger.warning("Access denied getting open file count."); open_files = "Access Denied"
        except Exception as e: logger.warning(f"Could not get open file count: {e}"); open_files = "Error"

        try: thread_count = process.num_threads()
        except psutil.AccessDenied: logger.warning("Access denied getting thread count."); thread_count = "Access Denied"
        except Exception as e: logger.warning(f"Could not get thread count: {e}"); thread_count = "Error"

        resources = {
            "memory": {"total_gb": round(mem.total/(1024**3),2), "available_gb": round(mem.available/(1024**3),2), "percent": mem.percent},
            "cpu_percent": cpu_percent,
            "process": {"pid": process.pid, "memory_used_mb": round(process_mem.rss/(1024**2),2), "open_files": open_files, "thread_count": thread_count}
        }
        logger.debug(f"System resources collected: {resources}")
        return resources
    except ImportError:
        logger.warning("psutil not installed, cannot report system resources.")
        return {"error": "psutil not installed"}
    except Exception as e:
        logger.error(f"Error collecting system resources: {str(e)}", exc_info=True)
        return {"error": f"Could not collect system resources: {str(e)}"}

# --- Detailed Diagnostics Function ---
def perform_detailed_diagnostics(logger: logging.Logger):
    """
    Performs detailed health checks for the environment and returns results.
    Uses the provided logger instance for logging.
    """
    logger.info("--- Starting Detailed Diagnostics ---")
    health_data = {
        "check_timestamp": datetime.now().isoformat(),
        "overall_status": "pending",
        "environment_details": {},
        "checks": {},
        "notes": []
    }

    # --- 1. NIX Chromium Check ---
    logger.debug("Diagnostic Check 1: NIX Chromium")
    nix_chromium_working = False
    nix_chromium_output = "Test not run or failed."
    nix_chromium_path = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH")
    original_launch_method = None

    health_data["environment_details"]["nix_chromium_path"] = nix_chromium_path if nix_chromium_path else "not set"

    if nix_chromium_path and os.path.exists(nix_chromium_path):
        logger.info(f"Attempting NIX Chromium test using path: {nix_chromium_path}")
        try:
            from playwright.sync_api import sync_playwright
            from playwright._impl._browser_type import BrowserType

            original_launch_method = BrowserType.launch
            logger.debug("Original Playwright launch method captured for diagnostic check.")

            # Define the patch specific to this diagnostic test
            def diagnostic_patched_launch(self, **kwargs):
                logger.debug(f"Diagnostic Patch: Forcing executablePath={nix_chromium_path} and headless=True")
                kwargs['executable_path'] = nix_chromium_path
                kwargs['headless'] = True
                current_env = os.environ.copy()
                current_env["PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS"] = "true"
                if "LD_LIBRARY_PATH" in os.environ:
                     current_env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
                kwargs['env'] = current_env
                if original_launch_method:
                    return original_launch_method(self, **kwargs)
                else:
                    logger.error("Diagnostic Patch: Original launch method not captured!")
                    raise RuntimeError("Original Playwright launch method not captured.")

            BrowserType.launch = diagnostic_patched_launch
            logger.debug("Diagnostic check: Playwright monkey patch applied.")

            with sync_playwright() as p:
                logger.debug("Launching Chromium via patched method...")
                browser = p.chromium.launch()
                version = browser.version
                logger.info(f"Browser launched. Version: {version}")
                page = browser.new_page()
                logger.debug("Navigating to example.com...")
                page.goto("https://example.com", timeout=30000)
                title = page.title()
                logger.info(f"Page loaded. Title: '{title}'")
                page.close()
                browser.close()
                logger.info("Browser closed.")
                nix_chromium_working = True
                nix_chromium_output = f"Success. Version: {version}. Loaded page title: '{title}'."
                logger.info("NIX Chromium test successful.")

        except ImportError as e:
            nix_chromium_working = False; nix_chromium_output = f"Import Error: {e}. Playwright installed?"; logger.error(f"ImportError during NIX Chromium test: {e}", exc_info=True)
        except Exception as e:
            nix_chromium_working = False; nix_chromium_output = f"Test Failed: {e}"; logger.error(f"Exception during NIX Chromium test: {e}", exc_info=True)
        finally:
            if original_launch_method:
                try: BrowserType.launch = original_launch_method; logger.debug("Restored original Playwright launch method.")
                except Exception as e: logger.warning(f"Failed to restore original launch method: {e}")

    elif not nix_chromium_path: nix_chromium_output = "Var PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH not set."; logger.warning(nix_chromium_output); health_data["notes"].append(nix_chromium_output)
    else: nix_chromium_output = f"Path '{nix_chromium_path}' set but file doesn't exist."; logger.warning(nix_chromium_output); health_data["notes"].append(nix_chromium_output)

    health_data["checks"]["nix_chromium"] = {"status": "success" if nix_chromium_working else "failure", "output": nix_chromium_output}
    if not nix_chromium_working: health_data["notes"].append("NIX Chromium test failed.")

    # --- 2. System Resources ---
    logger.debug("Diagnostic Check 2: System Resources")
    health_data["checks"]["system_resources"] = get_system_resources(logger) # Pass logger

    # --- 3. API Key Check ---
    logger.debug("Diagnostic Check 3: OpenAI API Key")
    api_key = os.getenv("OPENAI_API_KEY")
    api_key_status = "available" if api_key and api_key != "your_openai_api_key_here" else "missing_or_default"
    health_data["checks"]["openai_api_key"] = {"status": api_key_status}
    if api_key_status != "available": health_data["notes"].append("OpenAI API Key missing/default.")

    # --- 4. Python Environment ---
    logger.debug("Diagnostic Check 4: Python Environment")
    health_data["environment_details"]["python"] = {"version": sys.version, "executable": sys.executable, "platform": platform.platform()}

    # --- 5. Playwright Environment Variables ---
    logger.debug("Diagnostic Check 5: Playwright Env Vars")
    playwright_env_vars = {
        "PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS": os.environ.get("PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS"),
        "PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH": nix_chromium_path,
        "BROWSER_USE_BROWSER_ARGS": os.environ.get("BROWSER_USE_BROWSER_ARGS"),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH")
    }
    health_data["environment_details"]["playwright_env_vars"] = {k: (v if v is not None else "not set") for k, v in playwright_env_vars.items()}
    chromium_perms = "n/a"
    if nix_chromium_path and os.path.exists(nix_chromium_path):
        try: chromium_perms = "executable" if os.access(nix_chromium_path, os.X_OK) else "not_executable"
        except Exception as e: chromium_perms = f"check_failed: {e}"
    elif nix_chromium_path: chromium_perms = "path_does_not_exist"
    else: chromium_perms = "path_not_set"
    health_data["environment_details"]["playwright_env_vars"]["NIX_CHROMIUM_PERMISSIONS"] = chromium_perms


    # --- 6. Installed Packages Check ---
    logger.debug("Diagnostic Check 6: Installed Packages")
    packages_status = {}
    try:
        import playwright
        packages_status["playwright"] = { "status": "installed", "version": getattr(playwright, "__version__", "unknown") }
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p: packages_status["playwright"]["browsers"] = { "chromium": "available" if hasattr(p, "chromium") else "missing" }
        except Exception as browser_error: packages_status["playwright"]["browsers_check_error"] = str(browser_error); logger.warning(f"Error checking Playwright browser availability: {browser_error}")
    except ImportError as import_error: packages_status["playwright"] = { "status": "missing", "error": str(import_error) }; health_data["notes"].append("Playwright package not found.")
    try:
        import browser_use
        packages_status["browser_use"] = { "status": "installed", "version": getattr(browser_use, "__version__", "unknown") }
        try:
            agent_params = inspect.signature(browser_use.Agent.__init__).parameters; param_names = list(agent_params.keys()); param_names.remove('self')
            packages_status["browser_use"]["agent_init_params"] = param_names
        except Exception as param_error: packages_status["browser_use"]["agent_init_error"] = str(param_error); logger.warning(f"Error inspecting browser_use.Agent params: {param_error}")
    except ImportError as import_error: packages_status["browser_use"] = { "status": "missing", "error": str(import_error) }; health_data["notes"].append("browser-use package not found.")
    try: import psutil; packages_status["psutil"] = {"status": "installed", "version": getattr(psutil, "__version__", "unknown")}
    except ImportError as import_error: packages_status["psutil"] = {"status": "missing", "error": str(import_error)}; health_data["notes"].append("psutil package not found.")
    health_data["checks"]["installed_packages"] = packages_status


    # --- 7. System Dependencies Check (Linux Only) ---
    logger.debug("Diagnostic Check 7: System Dependencies (Linux)")
    system_deps = {}
    if platform.system() == "Linux":
        libs_to_check = [ "libnss3.so", "libnssutil3.so", "libsmime3.so", "libnspr4.so", "libdbus-1.so", "libatk-1.0.so", "libatk-bridge-2.0.so", "libatspi.so", "libcups.so", "libdrm.so", "libexpat.so", "libxcb.so", "libX11.so", "libXcomposite.so", "libXcursor.so", "libXdamage.so", "libXext.so", "libXfixes.so", "libXi.so", "libXrandr.so", "libXrender.so", "libXss.so", "libXtst.so", "libgbm.so", "libpango-1.0.so", "libfontconfig.so", "libfreetype.so", "libgcc_s.so", "libstdc++.so", "libxshmfence.so" ]
        try:
            logger.debug("Running 'ldconfig -p' to check system libraries...")
            result = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                ldconfig_output = result.stdout; found_count = 0
                for lib in libs_to_check:
                    if lib in ldconfig_output: system_deps[lib] = "found"; found_count += 1
                    else: system_deps[lib] = "not_found"
                logger.info(f"Found {found_count}/{len(libs_to_check)} potential system dependencies via ldconfig.")
            else: error_msg = f"'ldconfig -p' failed: {result.stderr}"; system_deps["error"] = error_msg; logger.error(error_msg); health_data["notes"].append("Failed ldconfig check.")
        except FileNotFoundError: error_msg = "'ldconfig' not found."; system_deps["error"] = error_msg; logger.warning(error_msg); health_data["notes"].append(error_msg)
        except subprocess.TimeoutExpired: error_msg = "'ldconfig -p' timed out."; system_deps["error"] = error_msg; logger.warning(error_msg); health_data["notes"].append(error_msg)
        except Exception as e: error_msg = f"Error running 'ldconfig -p': {e}"; system_deps["error"] = error_msg; logger.error(error_msg, exc_info=True); health_data["notes"].append("Error checking ldconfig.")
    else:
        system_deps["status"] = f"Skipped (Platform: {platform.system()})"
    health_data["checks"]["system_dependencies"] = system_deps


    # --- 8. Final Status Determination ---
    logger.debug("Determining Overall Status for Diagnostics")
    all_clear = all([
        nix_chromium_working,
        api_key_status == "available",
        packages_status.get("playwright", {}).get("status") == "installed",
        packages_status.get("browser_use", {}).get("status") == "installed",
        packages_status.get("psutil", {}).get("status") == "installed"
    ])
    health_data["overall_status"] = "healthy" if all_clear else "unhealthy"

    logger.info(f"--- Detailed Diagnostics Finished. Overall Status: {health_data['overall_status']} ---")
    if health_data["notes"]:
        logger.warning("Diagnostic Notes:\n- " + "\n- ".join(health_data["notes"]))

    return health_data
