# Project Name (Replace)

A brief description of your project.

## Setup & Running

1.  **Environment:** This project uses Poetry for Python dependency management within a Replit environment configured via Nix. The Node.js frontend/proxy and Python Flask backend are run concurrently.

2.  **Installation (Important: Manual Steps Required):**
    *   **Open the Shell tab.**
    *   When reloading the shell or environment, if prompted "Install Replit's Python tools [y/n]", press **'n' (no)**. This project uses Poetry to manage dependencies, and allowing Replit's tools to manage them simultaneously can cause conflicts.
    *   Run `poetry install` in the Shell tab. This installs Python dependencies (including `langchain-openai`, `langchain-google-genai`, `browser-use`, etc.) into the local `.pythonlibs` directory, based on `poetry.lock`.
    *   Run `npm install` in the Shell tab to install Node.js dependencies into `node_modules`.
    *   *(Note: The Replit "Run" button's automatic installation has been disabled in `.replit` due to previous issues. Manual installation via the Shell is required.)*

3.  **Environment Variables:**
    *   Copy `.env.example` to `.env`.
    *   Fill in your `OPENAI_API_KEY` (required for OpenAI models and default fallback).
    *   Fill in your `GOOGLE_API_KEY` (required if using Gemini models).
    *   **API Key:** Generate a secure secret key (e.g., a UUID) and set it for both `EXTERNAL_API_KEY` and `VITE_EXTERNAL_API_KEY`.
        *   `EXTERNAL_API_KEY`: Used by the Python backend (`api/auth.py`) to verify incoming requests to protected endpoints.
        *   `VITE_EXTERNAL_API_KEY`: Exposed specifically to the frontend build process (via Vite) so the UI (`client/src/lib/queryClient.ts`) can send the key in its requests to the backend. **Both variables in `.env` should have the same secret value.**
    *   Verify the `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` is correct for your Replit environment (run `which chromium | cat` in the shell if needed).

4.  **Running:** 
    *   Click the **"Run" button** in the Replit UI.
    *   This now executes the `dev` script defined in `package.json`.
    *   The `dev` script uses `concurrently` to start all three services:
        *   `npm run client`: Starts the Vite development server for the React frontend.
        *   `npm run server`: Starts the Node.js Express server (proxy).
        *   `npm run api`: Starts the Python Flask API backend using the Python interpreter within `.pythonlibs` (`./.pythonlibs/bin/python -m api.app`).
    *   You can view the frontend UI in the "Webview" tab.

## Browser Automation Notes

*   This project is configured to use the Chromium browser provided by the Nix environment (`replit.nix`) instead of downloading one via Playwright.
*   Playwright's browser launch mechanism is monkey-patched in `api/app.py` to force the use of the Nix Chromium executable and ensure it runs in headless mode, which is necessary for Replit's environment.

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
