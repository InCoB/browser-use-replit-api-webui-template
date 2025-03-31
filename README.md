# Project Name (Replace)

A brief description of your project.

## Setup & Running

1.  **Environment:** This project uses Poetry for Python dependency management within a Replit environment configured via Nix. The Node.js frontend/proxy and Python Flask backend are run concurrently.
2.  **Installation:**
    *   When reloading the shell or environment, if prompted "Install Replit's Python tools [y/n]", press **'n' (no)**. This project uses Poetry to manage dependencies, and allowing Replit's tools to manage them simultaneously can cause conflicts.
    *   Run `poetry install` in the Shell tab to install Python dependencies based on `poetry.lock`.
    *   Run `npm install` in the Shell tab to install Node.js dependencies.
3.  **Environment Variables:**
    *   Copy `.env.example` to `.env`.
    *   Fill in your `OPENAI_API_KEY`.
    *   **API Key:** Generate a secure secret key (e.g., a UUID) and set it for both `EXTERNAL_API_KEY` and `VITE_EXTERNAL_API_KEY`.
        *   `EXTERNAL_API_KEY`: Used by the Python backend (`api/auth.py`) to verify incoming requests to protected endpoints.
        *   `VITE_EXTERNAL_API_KEY`: Exposed specifically to the frontend build process (via Vite) so the UI (`client/src/lib/queryClient.ts`) can send the key in its requests to the backend. **Both variables in `.env` should have the same secret value.**
    *   Verify the `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` is correct for your Replit environment (run `which chromium | cat` in the shell if needed).
4.  **Running:** Use the "Run" button in Replit, which executes the `dev` script in `package.json`. This script uses `concurrently` to start both the Node.js server (`tsx server/index.ts`) and the Python API backend (`poetry run python api/app.py`).

## Browser Automation Notes

*   This project is configured to use the Chromium browser provided by the Nix environment (`replit.nix`) instead of downloading one via Playwright.
*   Playwright's browser launch mechanism is monkey-patched in `api/app.py` to force the use of the Nix Chromium executable and ensure it runs in headless mode, which is necessary for Replit's environment.

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
