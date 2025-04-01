import os
import sys
import asyncio
import traceback
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def print_environment_info():
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")
    print("\nEnvironment variables:")
    for key in ['LD_LIBRARY_PATH', 'PLAYWRIGHT_BROWSERS_PATH', 'BROWSER_USE_BROWSER_TYPE', 
                'BROWSER_USE_HEADLESS', 'PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS',
                'PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH']:
        print(f"{key}: {os.environ.get(key, 'Not set')}")

print_environment_info()

# Check for browser-use import
try:
    import browser_use
    print(f"\nbrowser-use version: {getattr(browser_use, '__version__', 'unknown')}")
except ImportError as e:
    print(f"\nFailed to import browser_use: {e}")

# Define a context manager for Playwright monkey patching
from playwright.sync_api import sync_playwright
from playwright._impl._browser_type import BrowserType

class BrowserPatch:
    def __init__(self, executable_path):
        self.executable_path = executable_path
        self.original_launch = None

    def __enter__(self):
        if not self.executable_path or not os.path.exists(self.executable_path):
            print(f"WARNING: PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH not set or file not found: {self.executable_path}")
            print("Patch might not work correctly.")
        if self.original_launch is None:
            self.original_launch = BrowserType.launch
            print("Stored original Playwright launch method.")

        def patched_launch(self_, **kwargs):
            print(f"Patched Playwright launch called, forcing executablePath={self.executable_path} and headless=True")
            if self.executable_path:
                kwargs['executablePath'] = self.executable_path
            else:
                print("WARNING: Cannot force executablePath, PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH is invalid.")
            kwargs['headless'] = True
            if 'env' not in kwargs or kwargs['env'] is None:
                kwargs['env'] = {}
            kwargs['env']['PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS'] = 'true'
            return self.original_launch(self_, **kwargs)

        BrowserType.launch = patched_launch
        print("Playwright monkey patch applied successfully.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_launch:
            BrowserType.launch = self.original_launch
            print("\nRestored original Playwright launch method.")

# Custom action: Save output to a file
def save_to_file(content, filename="agent_output.json"):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Output successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving output to file: {e}")

# Get the Chromium executable path from environment
chromium_path = os.environ.get('PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH')

# LLM Integration: Import ChatGoogleGenerativeAI from langchain_google_genai
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError as e:
    print(f"Failed to import langchain_google_genai: {e}")
    ChatGoogleGenerativeAI = None

# Get your GOOGLE_API_KEY from environment
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key or google_api_key.startswith("your_"):
    raise ValueError("GOOGLE_API_KEY not set or is a placeholder in environment variables.")

# Instantiate main LLM and planner LLM (using different models)
main_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
planner_llm = ChatGoogleGenerativeAI(model="o3-mini", google_api_key=google_api_key)

# Use BrowserPatch to manage Playwright's monkey patch
with BrowserPatch(chromium_path):
    try:
        with sync_playwright() as p:
            print("Playwright initialized successfully.")
            # Example multi-tab demonstration
            print("\nTesting multi-tab management:")
            try:
                browser = p.chromium.launch()
                context = browser.new_context()
                page1 = context.new_page()
                page1.goto('https://www.lonelyplanet.com')
                title1 = page1.title()
                print(f"  Tab 1 (Lonely Planet) loaded with title: {title1}")

                page2 = context.new_page()
                page2.goto('https://www.tripadvisor.com')
                title2 = page2.title()
                print(f"  Tab 2 (TripAdvisor) loaded with title: {title2}")

                page1.close()
                page2.close()
                browser.close()
                print("Multi-tab management test completed successfully!")
            except Exception as e:
                print("Multi-tab management test ERROR:")
                print(traceback.format_exc())
    except Exception as e:
        print("Error during Playwright setup or tests:")
        print(traceback.format_exc())

# Integrate LLM with browser-use Agent with full feature set
print("\nTesting browser-use Agent with full feature integration:")

try:
    from browser_use import Agent, BrowserConfig

    if not chromium_path or not os.path.exists(chromium_path):
        print(f"ERROR: Cannot run browser-use test, NIX Chromium path is invalid: {chromium_path}")
    elif ChatGoogleGenerativeAI is None:
        print("ERROR: LLM integration is unavailable because langchain_google_genai could not be imported.")
    else:
        print("Using NIX Chromium via patched Playwright with vision enabled and planner model.")
        # Create a BrowserConfig instance (you can further customize browser settings if needed)
        browser_config = BrowserConfig(headless=True)

        # Create an Agent with full features based on official documentation
        agent = Agent(
            task=(
                "Act as an expert travel planner. Navigate to https://www.lonelyplanet.com and use both textual and visual cues to extract destination hints. "
                "Then generate a creative weekend getaway itinerary that includes recommended flights, hotels, local attractions, and personalized travel tips. "
                "Output the final itinerary in JSON format."
            ),
            llm=main_llm,
            planner_llm=planner_llm,
            use_vision=True,                     # Enable vision capabilities for the main agent
            use_vision_for_planner=False,        # Disable vision for the planner to reduce costs
            planner_interval=4,                  # Plan every 4 steps
            save_conversation_path="logs/conversation",  # Save conversation history for debugging
            browser_config=browser_config
        )

        async def run_agent():
            print("Running browser-use Agent with full feature integration...")
            history = await agent.run()
            # Display some history details (optional)
            print("Visited URLs:", history.urls())
            print("Screenshots:", history.screenshots())
            print("Final result:", history.final_result())
            # Save full output
            save_to_file(history.final_result())

        asyncio.run(run_agent())
        print("Browser-use Agent test with full feature integration completed successfully!")
except ImportError as e:
    print(f"Failed to import browser-use components for test: {e}")
except Exception as e:
    print("Error running browser-use test with full feature integration:")
    print(traceback.format_exc())