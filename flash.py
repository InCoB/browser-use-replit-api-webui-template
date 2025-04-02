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
for key in [
    "LD_LIBRARY_PATH",
    "PLAYWRIGHT_BROWSERS_PATH",
    "BROWSER_USE_BROWSER_TYPE",
    "BROWSER_USE_HEADLESS",
    "PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS",
    "PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH",
]:
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
    print("\nApplying Playwright monkey patch and checking browser installation...")
    from playwright.sync_api import sync_playwright
    from playwright._impl._browser_type import BrowserType

    # --- Apply monkey patch ONCE here ---
    chromium_path = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH")
    if not chromium_path or not os.path.exists(chromium_path):
        print(
            f"WARNING: PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH not set or file not found: {chromium_path}"
        )
        print("Patch might not work correctly.")

    try:
        # Store the original launch method if not already stored
        if original_launch is None:
            original_launch = BrowserType.launch
            print("Stored original Playwright launch method.")

        # Define a patched launch method that forces NIX Chromium and headless mode
        def patched_launch(self, **kwargs):
            global original_launch  # Ensure we are using the globally stored original method
            print(
                f"Patched Playwright launch called, forcing executablePath={chromium_path} and headless=True"
            )

            # Force the executable path to NIX Chromium using correct parameter name
            if chromium_path:  # Only force if path is valid
                kwargs["executablePath"] = chromium_path
            else:
                print(
                    "WARNING: Cannot force executablePath, PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH is invalid."
                )

            # Force headless mode to True
            kwargs["headless"] = True

            # Ensure env is properly initialized
            if "env" not in kwargs or kwargs["env"] is None:
                kwargs["env"] = {}

            # Add skip validation flag if env is a dictionary
            if isinstance(kwargs["env"], dict):
                kwargs["env"]["PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS"] = "true"

            # Call the original stored launch method
            if original_launch:
                return original_launch(self, **kwargs)
            else:
                # Fallback if original_launch wasn't captured (should not happen)
                print("ERROR: Original launch method not found!")
                raise RuntimeError("Original Playwright launch method not captured.")

        # Apply the patch
        BrowserType.launch = patched_launch
        print("Playwright monkey patch applied successfully.")
    except Exception as e:
        print(f"Failed to apply Playwright monkey patch: {e}")
        sys.exit(1)  # Exit if patching failed, tests below will likely fail

    # --- End of monkey patch section ---

    with sync_playwright() as p:
        print("Playwright initialized successfully.")

        # Test Chromium using the patch (should force executable and headless)
        print("\nTesting Patched Chromium launch:")
        try:
            # No need to specify executablePath or headless=True here, patch handles it
            browser = p.chromium.launch()
            version = browser.version
            context = browser.new_context()
            page = context.new_page()
            page.goto("https://example.com")
            title = page.title()
            print(f"  Successfully loaded page with title: {title}")
            page.close()
            browser.close()
            print(f"  Patched Chromium: OK - version {version}")
        except Exception as e:
            print(f"  Patched Chromium: ERROR - {e}")
            import traceback

            print(f"  Detailed error for Patched Chromium:")
            print(f"  {traceback.format_exc()}")

        # Explicitly test with NIX Chromium path (redundant but good sanity check)
        print(
            "\nTesting Explicit NIX Chromium executable path (patch should handle this):"
        )
        try:
            if chromium_path and os.path.exists(chromium_path):
                print(f"  Attempting launch (patch forces path: {chromium_path})")
                # Patch should override executablePath, but we pass it anyway for clarity
                # Patch will also force headless=True
                browser = p.chromium.launch()
                version = browser.version
                context = browser.new_context()
                page = context.new_page()
                page.goto("https://example.com")
                title = page.title()
                print(f"  Successfully loaded page with title: {title}")
                browser.close()
                print(f"  Explicit NIX Chromium: OK - version {version}")
            else:
                print(
                    f"  Skipping test: PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH not valid ({chromium_path})"
                )
        except Exception as e:
            print(f"  Explicit NIX Chromium: ERROR - {e}")
            import traceback

            print(f"  Detailed error for Explicit NIX Chromium:")
            print(f"  {traceback.format_exc()}")

except ImportError as e:
    print(f"Failed to import playwright: {e}")
except Exception as e:
    print(f"Error during Playwright setup or tests: {e}")
    import traceback

    print(f"Detailed error:")
    print(f"{traceback.format_exc()}")

# Finally, test browser-use (should use the globally patched launch)
print("\nTesting browser-use with globally patched Playwright:")
try:
    # No need to re-patch here, the patch applied earlier should be active
    import browser_use
    from browser_use import Agent, BrowserConfig
    import asyncio

    # Remove OpenAI import if not used elsewhere, add Google import
    # from langchain_openai import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not chromium_path or not os.path.exists(chromium_path):
        print(
            f"ERROR: Cannot run browser-use test, NIX Chromium path is invalid: {chromium_path}"
        )
    else:
        print(f"Using NIX Chromium via patched Playwright launch.")

        # Check what parameters BrowserConfig accepts
        import inspect

        browser_config_params = inspect.signature(BrowserConfig.__init__).parameters
        print(f"BrowserConfig parameters: {list(browser_config_params.keys())}")

        # Create browser config - patch will force executablePath and headless
        # Setting headless=True here is good practice but patch ensures it
        browser_config = BrowserConfig(headless=True)

        # Check what parameters Agent accepts
        agent_params = inspect.signature(Agent.__init__).parameters
        print(f"Agent parameters: {list(agent_params.keys())}")

        # Create an agent instance
        agent = None
        llm_instance = None

        # Determine how to instantiate the agent based on available parameters
        if "browser_config" in agent_params:
            # Preferred method: pass the browser_config
            print("Instantiating Agent with browser_config.")
            agent = Agent(
                task="Go to example.com and tell me the title of the page",
                browser_config=browser_config,
                # NOTE: Even with browser_config, llm might still be required depending on Agent version
                # If you still get errors, you might need to add an LLM here too.
            )
        else:
            # Fallback: Instantiate with the intended LLM (Gemini Flash)
            print(
                "Instantiating Agent with Gemini Flash LLM (fallback as browser_config not detected or Agent requires LLM anyway)."
            )
            llm_instance = None
            agent = None  # Initialize agent to None
            try:
                # Ensure Google API key is set
                google_api_key = os.getenv("GOOGLE_API_KEY")
                if not google_api_key or google_api_key.startswith("your_"):
                    raise ValueError(
                        "GOOGLE_API_KEY not found or is a placeholder in environment variables."
                    )
                # Use the desired Gemini Flash model for the test
                llm_instance = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash", google_api_key=google_api_key
                )
                agent = Agent(
                    task="go to google, find recent AI news at least 10 different articles and make a digest for me, if it mentions any project go this project and gather info about it",
                    llm=llm_instance,
                )
            except ImportError:
                print(
                    "ERROR: langchain_google_genai not installed. Cannot create Agent with Gemini LLM."
                )
            except ValueError as ve:
                print(f"ERROR: {ve}")
            except Exception as llm_err:
                print(f"ERROR: Failed to instantiate Gemini LLM or Agent: {llm_err}")

        # Run the agent if instantiation was successful
        if agent:

            async def run_agent():
                print("Running browser-use agent...")
                result = await agent.run()
                print(f"Browser-use agent result: {result}")

            asyncio.run(run_agent())
            print("Browser-use test completed successfully!")
        else:
            print("Could not instantiate browser_use Agent.")

except ImportError as e:
    print(f"Failed to import browser-use components for test: {e}")
except Exception as e:
    print(f"Error running browser-use test: {e}")
    import traceback

    print(f"Detailed error for browser-use test:")
    print(f"{traceback.format_exc()}")

# Optional: Restore original launch method at the very end
if original_launch:
    try:
        from playwright._impl._browser_type import BrowserType

        BrowserType.launch = original_launch
        print("\nRestored original Playwright launch method.")
    except Exception as e:
        print(f"\nFailed to restore original Playwright launch method: {e}")
