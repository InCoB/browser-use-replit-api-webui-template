#!/usr/bin/env python3
"""Test script to validate if the patching approach works with NIX Chromium."""

import os
import sys
import traceback

# Set environment variables
os.environ["PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS"] = "1"
os.environ["PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD"] = "1"
os.environ["PLAYWRIGHT_CHROMIUM_SKIP_SYSTEM_DEPS"] = "true"
os.environ["BROWSER_USE_BROWSER_TYPE"] = "chromium"
os.environ["BROWSER_USE_HEADLESS"] = "true"
os.environ["PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH"] = "/nix/store/zi4f80l169xlmivz8vja8wlphq74qqk0-chromium-125.0.6422.141/bin/chromium"

# Store original launch method
original_launch = None

# Print environment
print(f"Python version: {sys.version}")
print(f"Chromium path: {os.environ.get('PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH')}")

try:
    # Import playwright
    import playwright
    from playwright.sync_api import sync_playwright
    print(f"Playwright version: {getattr(playwright, '__version__', 'unknown')}")
    
    # Import BrowserType for patching
    try:
        from playwright._impl._browser_type import BrowserType
        
        # Store original launch method
        original_launch = BrowserType.launch
        print("Original launch method captured")
        
        # Define patched launch method
        def patched_launch(self, **kwargs):
            print("Patched launch method called")
            chromium_path = os.environ.get('PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH')
            print(f"Setting executablePath={chromium_path}")
            
            # Force executable path to NIX Chromium
            kwargs['executablePath'] = chromium_path
            
            # Force headless mode
            kwargs['headless'] = True
            
            # Ensure env is set up
            if 'env' not in kwargs or kwargs['env'] is None:
                kwargs['env'] = {}
            
            # Add skip validation flag
            if isinstance(kwargs['env'], dict):
                kwargs['env']['PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS'] = '1'
            
            # Call original launch method
            print(f"Calling original launch with kwargs: {kwargs}")
            return original_launch(self, **kwargs)
        
        # Apply patch
        BrowserType.launch = patched_launch
        print("Playwright patched successfully")
    except Exception as e:
        print(f"Failed to patch Playwright: {e}")
        traceback.print_exc()
    
    # Test with sync_playwright
    with sync_playwright() as p:
        print("Playwright initialized successfully")
        
        # Test patched chromium
        try:
            print("Launching Chromium browser...")
            browser = p.chromium.launch()
            print(f"Browser launched, version: {browser.version}")
            
            # Create context and page
            context = browser.new_context()
            page = context.new_page()
            
            # Navigate to a page
            print("Navigating to example.com...")
            page.goto('https://example.com')
            
            # Get page title
            title = page.title()
            print(f"Page title: {title}")
            
            # Close browser
            page.close()
            browser.close()
            print("Test completed successfully")
        except Exception as e:
            print(f"Error testing Chromium: {e}")
            traceback.print_exc()
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
    traceback.print_exc()