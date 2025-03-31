"""Check browser installation and accessibility."""

import os
import sys
from pathlib import Path

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Working directory: {os.getcwd()}")

# Print environment variables
print("\nEnvironment variables:")
for key in ['LD_LIBRARY_PATH', 'PLAYWRIGHT_BROWSERS_PATH', 'BROWSER_USE_BROWSER_TYPE', 
           'BROWSER_USE_HEADLESS', 'PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS']:
    print(f"{key}: {os.environ.get(key, 'Not set')}")

# Check for browser-use
try:
    import browser_use
    print(f"\nbrowser-use version: {getattr(browser_use, '__version__', 'unknown')}")
except ImportError as e:
    print(f"\nFailed to import browser_use: {e}")

# Check for playwright
try:
    import playwright
    print(f"playwright version: {getattr(playwright, '__version__', 'unknown')}")
    
    # Check for browser installation
    print("\nChecking browser installation...")
    from playwright.sync_api import sync_playwright
    
    with sync_playwright() as p:
        print("Playwright initialized successfully")
        for browser_type in ['chromium', 'firefox', 'webkit']:
            if hasattr(p, browser_type):
                try:
                    browser = getattr(p, browser_type).launch(headless=True)
                    version = browser.version
                    browser.close()
                    print(f"  {browser_type}: OK - version {version}")
                except Exception as e:
                    print(f"  {browser_type}: ERROR - {e}")
            else:
                print(f"  {browser_type}: Not available")
                
except ImportError as e:
    print(f"Failed to import playwright: {e}")
except Exception as e:
    print(f"Error checking browsers: {e}")