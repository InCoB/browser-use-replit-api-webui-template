"""Check browser installation and accessibility.

This script will run various tests to check if browser automation works
with the NIX Chromium executable."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Store the original launch method globally so it can be accessed from any scope
original_launch = None

# Load environment variables from .env file
load_dotenv()

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Working directory: {os.getcwd()}")

# Print environment variables
print("\nEnvironment variables:")
for key in ['LD_LIBRARY_PATH', 'PLAYWRIGHT_BROWSERS_PATH', 'BROWSER_USE_BROWSER_TYPE', 
           'BROWSER_USE_HEADLESS', 'PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS',
           'PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH']:
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
    
    # Apply monkey patch for BrowserType.launch before initializing playwright
    chromium_path = os.environ.get('PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH')
    try:
        # Import the BrowserType class to apply monkey patch
        from playwright._impl._browser_type import BrowserType
        
        # Store the original launch method
        original_launch = BrowserType.launch
        
        # Define a patched launch method that forces the NIX Chromium executable
        def patched_launch(self, **kwargs):
            print(f"Patched Playwright launch called, forcing executablePath={chromium_path}")
            
            # Force the executable path to NIX Chromium using correct parameter name
            kwargs['executablePath'] = chromium_path
            
            # Ensure env is properly initialized
            if 'env' not in kwargs or kwargs['env'] is None:
                kwargs['env'] = {}
            
            # Add skip validation flag if env is a dictionary
            if isinstance(kwargs['env'], dict):
                kwargs['env']['PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS'] = 'true'
            
            return original_launch(self, **kwargs)
        
        # Apply the patch
        BrowserType.launch = patched_launch
        print("Playwright monkey patch applied successfully")
    except Exception as e:
        print(f"Failed to apply Playwright monkey patch: {e}")
    
    with sync_playwright() as p:
        print("Playwright initialized successfully")
        
        # Only test Chromium with our patched browser launch
        print("\nTesting monkey-patched Chromium:")
        try:
            browser = p.chromium.launch(headless=True)
            version = browser.version
            context = browser.new_context()
            page = context.new_page()
            page.goto('https://example.com')
            title = page.title()
            print(f"  Successfully loaded page with title: {title}")
            page.close()
            browser.close()
            print(f"  Chromium: OK - version {version}")
        except Exception as e:
            print(f"  Chromium: ERROR - {e}")
            # Print error details
            import traceback
            print(f"  Detailed error for Chromium:")
            print(f"  {traceback.format_exc()}")
        
        # Then try with the Nix Chromium executable path
        print("\nTesting with NIX Chromium executable path:")
        try:
            chromium_path = os.environ.get('PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH')
            if chromium_path:
                print(f"  Using Chromium at: {chromium_path}")
                # Check if file exists
                if os.path.exists(chromium_path):
                    print(f"  Chromium executable exists at path")
                    # Launch with explicit executable path using correct parameter name
                    browser = p.chromium.launch(
                        executablePath=chromium_path,
                        headless=True
                    )
                    version = browser.version
                    context = browser.new_context()
                    page = context.new_page()
                    page.goto('https://example.com')
                    title = page.title()
                    print(f"  Successfully loaded page with title: {title}")
                    browser.close()
                    print(f"  NIX Chromium: OK - version {version}")
                else:
                    print(f"  ERROR: Chromium executable not found at path")
            else:
                print(f"  PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH not set")
        except Exception as e:
            print(f"  NIX Chromium: ERROR - {e}")
            # Print error details
            import traceback
            print(f"  Detailed error for NIX Chromium:")
            print(f"  {traceback.format_exc()}")
                
except ImportError as e:
    print(f"Failed to import playwright: {e}")
except Exception as e:
    print(f"Error checking browsers: {e}")
    # Print error details
    import traceback
    print(f"Detailed error:")
    print(f"{traceback.format_exc()}")

# Finally, test browser-use with NIX Chromium directly
print("\nTesting browser-use with NIX Chromium:")
try:
    import browser_use
    from browser_use import Agent, BrowserConfig
    import asyncio
    
    chromium_path = os.environ.get('PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH')
    if not chromium_path or not os.path.exists(chromium_path):
        print(f"ERROR: Chromium executable not found at: {chromium_path}")
    else:
        print(f"Using Chromium at: {chromium_path}")
        
        # Apply monkey patch for Playwright's BrowserType for browser-use
        try:
            # Import the BrowserType class to apply the monkey patch
            from playwright._impl._browser_type import BrowserType
            
            # Check if we need to restore the original launch method
            if 'original_launch' in locals():
                # Original launch already saved, so we can just reapply the patch
                print("Re-applying Playwright monkey patch for browser-use")
            else:
                # First time patching, save the original launch method
                original_launch = BrowserType.launch
                print("Saved original launch method for later restoration")
                
            # Define a patched launch method that forces the NIX Chromium executable
            def patched_launch(self, **kwargs):
                print(f"browser-use: Patched Playwright launch forcing executablePath={chromium_path}")
                
                # Force the executable path to NIX Chromium using correct parameter name
                kwargs['executablePath'] = chromium_path
                
                # Ensure env is properly initialized
                if 'env' not in kwargs or kwargs['env'] is None:
                    kwargs['env'] = {}
                
                # Add skip validation flag if env is a dictionary
                if isinstance(kwargs['env'], dict):
                    kwargs['env']['PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS'] = 'true'
                
                return original_launch(self, **kwargs)
            
            # Apply the patch
            BrowserType.launch = patched_launch
            print("Playwright monkey patch applied for browser-use")
        except Exception as e:
            print(f"Failed to apply Playwright monkey patch for browser-use: {e}")
        
        # Check what parameters BrowserConfig accepts
        import inspect
        browser_config_params = inspect.signature(BrowserConfig.__init__).parameters
        print(f"BrowserConfig parameters: {list(browser_config_params.keys())}")
        
        # Create browser config for the NIX Chromium using the appropriate parameters
        # Create browser config with just the headless parameter
        # Our monkey patching will handle the executablePath
        browser_config = BrowserConfig(
            headless=True
        )
        
        # Check what parameters Agent accepts
        agent_params = inspect.signature(Agent.__init__).parameters
        print(f"Agent parameters: {list(agent_params.keys())}")
        
        # Create an agent with the custom browser config and appropriate parameters
        if 'browser_config' in agent_params:
            agent = Agent(
                task="Go to example.com and tell me the title of the page",
                browser_config=browser_config
            )
        elif 'llm' in agent_params:
            # We need to specify an LLM to use
            from langchain_openai import ChatOpenAI
            
            # Create an LLM instance using the OPENAI_API_KEY from environment variables
            llm = ChatOpenAI(model_name="gpt-4o")
            
            agent = Agent(
                task="Go to example.com and tell me the title of the page",
                llm=llm
            )
        else:
            # Fall back to minimal parameters
            agent = Agent(
                task="Go to example.com and tell me the title of the page"
            )
        
        # Run the agent synchronously
        async def run_agent():
            result = await agent.run()
            print(f"Browser-use agent result: {result}")
            
        # Run the async function
        print("Running browser-use agent...")
        asyncio.run(run_agent())
        print("Browser-use test completed successfully!")
        
except ImportError as e:
    print(f"Failed to import browser-use components: {e}")
except Exception as e:
    print(f"Error running browser-use test: {e}")
    # Print error details
    import traceback
    print(f"Detailed error for browser-use test:")
    print(f"{traceback.format_exc()}")