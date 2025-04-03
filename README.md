# Browser-Use Replit + API
https://github.com/browser-use/browser-use running in replit 

## Setup & Running

[replit link https://replit.com/@nksokolo/Browser-Use-Replit-Template?v=1]

1.  **Environment:** This project uses Poetry for Python dependency management within a Replit environment configured via Nix. The Node.js frontend/proxy and Python Flask backend are run concurrently.

2.  **Nix Environment Dependencies (Important):**
    *   System-level dependencies, including the core Python interpreter and specific Python packages available through Nix (like `psutil`, `black`), **must** be added to the `deps` list within the `replit.nix` file.
    *   After modifying `replit.nix`, you **must** rebuild the environment by running `kill 1` in the **Shell** tab. Poetry cannot install packages that are intended to be provided by the Nix environment.

3.  **Installation (Manual Steps Required):**
    *   **Open the Shell tab.**
    *   When reloading the shell or environment, if prompted "Install Replit's Python tools [y/n]", press **'n' (no)**.
    *   Run `poetry install`. This installs Python dependencies listed in `pyproject.toml` (like `langchain-openai`, `langchain-google-genai`, `browser-use`, etc.) into the local `.pythonlibs` directory using the Python interpreter provided by Nix.
    *   Run `npm install` to install Node.js dependencies.
    *   *(Note: The Replit \"Run\" button's automatic installation has been disabled in `.replit`.)*

4.  **Environment Variables:**
    *   Copy `.env.example` to `.env`.
    *   Fill in your `OPENAI_API_KEY` (required for OpenAI models and default fallback).
    *   Fill in your `GOOGLE_API_KEY` (required if using Gemini models).
    *   **API Key:** Generate a secure secret key (e.g., a UUID) and set it for both `EXTERNAL_API_KEY` and `VITE_EXTERNAL_API_KEY`.
        *   `EXTERNAL_API_KEY`: Used by the Python backend (`api/auth.py`) to verify incoming requests to protected endpoints.
        *   `VITE_EXTERNAL_API_KEY`: Exposed specifically to the frontend build process (via Vite) so the UI (`client/src/lib/queryClient.ts`) can send the key in its requests to the backend. **Both variables in `.env` should have the same secret value.** ---->>> PLEASE SEE KNOWN ISSUES AS THEY KEY IS HARDCODED
    *   Verify the `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` is correct for your Replit environment (run `which chromium | cat` in the shell if needed).///



5.  **Running:** 
    *   Click the **"Run" button** in the Replit UI.
    *   This now executes the `dev` script defined in `package.json`.
    *   The `dev` script uses `concurrently` to start all three services:
        *   `npm run client`: Starts the Vite development server for the React frontend.
        *   `npm run server`: Starts the Node.js Express server (proxy).
        *   `npm run api`: Starts the Python Flask API backend using the Python interpreter within `.pythonlibs` (`./.pythonlibs/bin/python -m api.app`).
    *   You can view the frontend UI in the "Webview" tab.

6. **Running without Server (No Server Required)**

This project includes standalone example scripts that demonstrate how to use the browser automation functionality directly, without launching the full server stack:

### Simple Browser Check

**examples/Simple.py**: A basic script to verify browser configuration and accessibility:

```bash
python examples/Simple.py
```

This script:
* Verifies environment variables and Python version
* Applies the Playwright monkey patch for NIX Chromium
* Tests browser launch, navigation, and basic interaction
* Diagnostics browser compatibility issues

### Advanced Browser Automation with Chat

**examples/chat_after_finish.py**: A comprehensive example with custom patches and interactive chat:

```bash
python examples/chat_after_finish.py
```

This script:
* Applies several patches to improve browser-use behavior:
  * Fixes memory initialization
  * Forces main LLM for planning
  * Adds error recovery and self-correction
* Adds custom JavaScript execution and iframe content extraction actions
* Runs a complex browser automation task
* Provides an interactive chat interface with the LLM after task completion

### Using Examples vs. Server API

**Direct Script Usage:**
* Faster to start and simpler for testing/experimentation
* Doesn't require multiple services to be running
* More control over specific implementation details
* Better for debugging browser-use functionality
* Allows interactive chat with LLM after task completion

**Server API Approach:**
* Provides a RESTful API for programmatic access
* Supports multiple clients and concurrent requests
* Better for production deployments
* Cleanly separates frontend and backend concerns
* More scalable for multiple users/applications



## Browser Automation Notes

*   This project is configured to use the Chromium browser provided by the Nix environment (`replit.nix`) instead of downloading one via Playwright.
*   Playwright's browser launch mechanism is monkey-patched conditionally in `api/app.py` when running in Replit to force the use of the Nix Chromium executable and ensure it runs in headless mode.
*   The `/diagnostics` endpoint verifies that the Nix Chromium executable can be launched successfully and reports whether the Replit environment (`REPL_ID`) is detected, which determines if the patch should be active during task execution.

## Logging

The Python API (`api/app.py`) uses Python's standard `logging` module.

*   **Output:** Logs are sent to both the Replit Console and a rotating file named `api.log` in the project root.
*   **Log File:** `api.log` will grow up to 5MB and keep up to 2 backup files (`api.log.1`, `api.log.2`). This file is included in `.gitignore`.
*   **Log Level:** The verbosity of the logs (for both console and file) is controlled by the `FLASK_LOG_LEVEL` environment variable set in your `.env` file.
    *   `INFO` (Default): Shows general progress, startup messages, task status changes, and errors.
    *   `DEBUG`: Shows much more detailed information, including function calls, variable values, Playwright patch steps, API key checks, etc. Useful for deep debugging.
    *   `WARNING`, `ERROR`, `CRITICAL`: Show progressively less information.

## Troubleshooting

*   **Dependency Issues:** If you encounter Python import errors, try cleaning the environment and reinstalling:
    ```bash
    rm -rf .pythonlibs
    poetry lock # Ensure lock file is consistent
    poetry install
    ```
    If you encounter Node.js errors, try removing `node_modules` and reinstalling:
    ```bash
    rm -rf node_modules
    npm install
    ```
*   **Browser Path:** If browser tasks fail, double-check the `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` in `.env` matches the output of `which chromium | cat` in the shell.
*   **Port Conflicts:** If you see "Address already in use", ensure only the `concurrently` script in `package.json` is starting the Python API (check that `server/routes.ts` doesn't also try to start it).

## API Authentication

*   The project uses `EXTERNAL_API_KEY` and `VITE_EXTERNAL_API_KEY` for API authentication.
*   Generate a secure secret key (e.g., a UUID) and set it for both variables in `.env`.
*   `EXTERNAL_API_KEY`: Used by the Python backend (`api/auth.py`) to verify incoming requests to protected endpoints.
*   `VITE_EXTERNAL_API_KEY`: Exposed specifically to the frontend build process (via Vite) so the UI (`client/src/lib/queryClient.ts`) can send the key in its requests to the backend. **Both variables in `.env` should have the same secret value.**

*   **Resource Limits:** Monitor Replit Console for crashes due to memory/CPU limits.

## Testing the API, Browser Automation, and Environment

The project includes a comprehensive testing script that verifies multiple components:

**main_test.py**: Tests the following:

**1. Environment Diagnostics:**
* Tests the `/diagnostics` endpoint 
* Verifies Nix Chromium installation and executable access
* Checks system resources (memory, CPU usage)
* Validates environment variables and API key availability
* Confirms Python environment and Playwright configuration

**2. Browser Automation:**
* Creates a real browser automation task using the configured Chromium
* Verifies browser launch, navigation, and content extraction
* Tests the complete browser lifecycle from initialization to cleanup

**3. API Functionality:**  
* Tests API authentication with the configured API key
* Verifies task creation, status polling, and result retrieval
* Validates error handling and response format
* Confirms thread and resource management

### Running the Test

1. **Ensure the main application is running** (click the "Run" button).
2. **Configure `.env`:** Make sure `API_BASE_URL` and `EXTERNAL_API_KEY` are correctly set in your `.env` file.
3. **Run the test script:** Open the **Shell** tab and execute:
   ```bash
   python main_test.py
   ```

### Expected Output
The test script produces a comprehensive report including:
* Detailed diagnostics about your environment configuration
* Browser execution status and compatibility information
* Raw API responses with task status transitions
* Final extracted content from the automated browser session

This comprehensive test ensures all components of the system are working correctly together, from environment configuration to browser automation to API communication.

## Known Issues ⚠️

*   **Hardcoded Frontend API Key:** Due to an unresolved issue with Vite correctly loading `.env` variables prefixed with `VITE_` within this specific Replit environment, the frontend API key is currently **hardcoded** in `client/src/components/demo-console.tsx`.
    *   **Action Required:** You **must** manually replace the placeholder key `'93ecb5a7-64f6-4d3c-9ba1-f5ca5eadc1f9'` inside the `fetchApi` and `postApi` functions in that file with the **same secret key** you set for `EXTERNAL_API_KEY` and `VITE_EXTERNAL_API_KEY` in your `.env` file.
    *   Look for the `// TODO: Fix Vite .env loading and remove hardcoded key` comments.
    *   Failure to do this will result in authentication errors when using the web UI.
*   **`RuntimeError: Event loop is closed`:** You may occasionally see this error in the Replit Console logs *after* a browser task completes successfully. This appears related to the cleanup of asynchronous network resources (like those used by `httpx` within underlying libraries) after the background task's event loop has finished. In this template's current configuration, it usually doesn't affect the task's successful execution or results, but indicates imperfect async resource management during shutdown.


