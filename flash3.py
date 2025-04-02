"""Check browser installation and accessibility using a global monkey patch
   AND patching the Agent._run_planner method to use main LLM (v2).
   Includes increased max_steps. Task set to fill interactive form.
"""

import os
import sys
import asyncio
import traceback
from pathlib import Path
from dotenv import load_dotenv
import inspect
import functools  # For wrapping (though not used in current patch)
import json  # Needed for logging plan in patch

# --- Early Imports ---
try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    print("Imported ChatGoogleGenerativeAI.")
except ImportError as e:
    print(f"Failed to import ChatGoogleGenerativeAI: {e}")
    sys.exit(1)

try:
    from playwright._impl._browser_type import BrowserType
    from playwright.sync_api import sync_playwright

    print("Imported Playwright components.")
except ImportError as e:
    print(f"Failed to import Playwright components: {e}")
    sys.exit(1)

try:
    # Import the Agent class itself for patching
    from browser_use import Agent

    # Try importing other components needed by patch
    from browser_use.agent.prompts import PlannerPrompt
    from browser_use.agent.message_manager.utils import convert_input_messages
    from browser_use.exceptions import LLMException

    print("Imported browser_use.Agent and patch dependencies.")
    AGENT_CLASS_AVAILABLE = True
except ImportError as e:
    print(
        f"WARN: Could not import browser_use.Agent or patch dependencies: {e}. Internal patch may not apply correctly."
    )
    AGENT_CLASS_AVAILABLE = False
# --- End Early Imports ---


# Store the original methods globally
original_playwright_launch = None
original_agent_run_planner = None  # To store original Agent method

# === Start browser_use Agent Patch for _run_planner ===
if AGENT_CLASS_AVAILABLE:
    target_method_name = "_run_planner"
    if hasattr(Agent, target_method_name):
        original_agent_run_planner = getattr(Agent, target_method_name)
        print(f"Stored original Agent method: {target_method_name}")

        async def patched_run_planner(self, *args, **kwargs):
            """Patched version to use main LLM instead of planner LLM."""
            global original_agent_run_planner
            print(
                f"DEBUG Patch: Intercepting {target_method_name}. FORCING MAIN LLM FOR PLANNING."
            )

            # Get the main LLM object directly from self
            main_llm = getattr(self, "llm", None)

            # Check if planning should even run
            if not self.settings.planner_llm or not main_llm:
                print(
                    "DEBUG Patch: Planner LLM not configured or Main LLM missing, skipping planner."
                )
                if not original_agent_run_planner:
                    raise RuntimeError("Original _run_planner not captured!")
                return await original_agent_run_planner(self, *args, **kwargs)

            # --- Execute planning using MAIN LLM ---
            print(
                f"DEBUG Patch: Attempting planning using MAIN LLM ({getattr(main_llm, 'model', '?')})."
            )
            try:
                # Prepare messages (copied logic from original _run_planner)
                page = await self.browser_context.get_current_page()
                standard_actions = self.controller.registry.get_prompt_description()
                page_actions = self.controller.registry.get_prompt_description(page)
                all_actions = standard_actions
                if page_actions:
                    all_actions += "\n" + page_actions

                planner_messages = [
                    PlannerPrompt(all_actions).get_system_message(
                        self.settings.is_planner_reasoning
                    ),
                    *self._message_manager.get_messages()[1:],
                ]

                # Handle vision removal for planner (copied logic)
                if (
                    not self.settings.use_vision_for_planner
                    and self.settings.use_vision
                ):
                    last_state_message = planner_messages[-1]
                    new_msg = ""
                    content_to_process = last_state_message.content
                    if isinstance(content_to_process, list):
                        for msg_item in content_to_process:
                            if (
                                isinstance(msg_item, dict)
                                and msg_item.get("type") == "text"
                            ):
                                new_msg += msg_item.get("text", "")
                    elif isinstance(content_to_process, str):
                        new_msg = content_to_process
                    if new_msg != content_to_process:
                        planner_messages[-1] = type(last_state_message)(content=new_msg)

                # Convert messages using main model name (copied logic)
                main_model_name = getattr(main_llm, "model", "Unknown")
                planner_messages = convert_input_messages(
                    planner_messages, main_model_name
                )

                # --- CORE PATCH: Invoke main_llm ---
                response = await main_llm.ainvoke(planner_messages)
                # ------------------------------------

                plan = str(response.content)

                # Handle think tags if necessary (copied logic)
                if main_model_name and (
                    "deepseek-r1" in main_model_name
                    or "deepseek-reasoner" in main_model_name
                ):
                    if hasattr(self, "_remove_think_tags"):
                        plan = self._remove_think_tags(plan)

                # Log plan (copied logic)
                log_func = getattr(self, "logger", None)
                info_func = getattr(log_func, "info", print) if log_func else print
                debug_func = getattr(log_func, "debug", print) if log_func else print
                try:
                    plan_json = json.loads(plan)
                    info_func(
                        f"Planning Analysis (via main LLM in patch):\n{json.dumps(plan_json, indent=4)}"
                    )
                except json.JSONDecodeError:
                    info_func(f"Planning Analysis (via main LLM in patch):\n{plan}")
                except Exception as e:
                    debug_func(f"Error parsing planning analysis: {e}")
                    info_func(f"Plan: {plan}")

                return plan  # Return the generated plan string

            except Exception as e:
                log_func = getattr(self, "logger", None)
                error_func = getattr(log_func, "error", print) if log_func else print
                error_func(f"Error during patched planning execution: {str(e)}")
                raise RuntimeError(f"Patched planning failed: {e}") from e

        # --- End Patched Method ---

        # Apply the patch to the class
        setattr(Agent, target_method_name, patched_run_planner)
        print(f"Applied patch to Agent method: {target_method_name}")
    else:
        print(
            f"WARN: Could not find method '{target_method_name}' on Agent class to patch. Internal patch not applied."
        )
else:
    print("WARN: browser_use.Agent not available. Internal patch not applied.")
# === End browser_use Agent Patch ===


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

# (Redundant checks for browser_use and playwright kept for structure)
try:
    import browser_use

    try:
        print(f"\nbrowser-use version: {browser_use.__version__}")
    except AttributeError:
        print("\nbrowser-use version: unknown")
except ImportError as e:
    print(f"\nFailed to import browser_use: {e}")
try:
    import playwright

    print(f"playwright version: {getattr(playwright, '__version__', 'unknown')}")
except ImportError as e:
    print(f"Failed to import playwright: {e}")


# --- Apply Playwright monkey patch ONCE here ---
print("\nApplying Playwright monkey patch...")
chromium_path = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH")
if not chromium_path or not os.path.exists(chromium_path):
    print(
        f"WARNING: PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH not set or file not found: {chromium_path}"
    )
    print("Playwright Patch might not work correctly.")
    chromium_path = None

try:
    if original_playwright_launch is None:
        original_playwright_launch = BrowserType.launch
        print("Stored original Playwright launch method.")

    # Define patched playwright launch (same as before)
    def patched_playwright_launch(self, **kwargs):
        global original_playwright_launch
        current_chromium_path = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH")
        print(
            f"Patched Playwright launch called, forcing executablePath={current_chromium_path} and headless=True"
        )
        if current_chromium_path and os.path.exists(current_chromium_path):
            kwargs["executablePath"] = current_chromium_path
        else:
            print(
                f"WARNING: Cannot force executablePath, invalid path: {current_chromium_path}"
            )
        kwargs["headless"] = True
        if "env" not in kwargs or kwargs["env"] is None:
            kwargs["env"] = {}
        if isinstance(kwargs["env"], dict):
            kwargs["env"]["PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS"] = "true"
        else:
            print(f"WARNING: Playwright launch 'env' not a dict.")
        if original_playwright_launch:
            return original_playwright_launch(self, **kwargs)
        else:
            raise RuntimeError("Original Playwright launch method not captured!")

    BrowserType.launch = patched_playwright_launch
    print("Playwright monkey patch applied successfully.")
except Exception as e:
    print(f"Failed to apply Playwright monkey patch: {e}")
    sys.exit(1)
# --- End Playwright monkey patch ---


# Define save_to_file function
def save_to_file(content, filename="agent_output.json"):
    try:
        output_dir = Path(filename).parent
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            # import json # Already imported in patch
            try:
                if isinstance(content, (dict, list)):
                    json.dump(content, f, indent=4, ensure_ascii=False)
                else:
                    f.write(str(content))
            except (TypeError, OverflowError):
                f.write(str(content))
        print(f"Output successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving output to file '{filename}': {e}")


# === Testing browser-use with Targeted Agent Patch ===
print("\nTesting browser-use with Patched Agent._run_planner:")
try:
    # Ensure Agent class is available if not imported earlier
    if "Agent" not in globals() and "browser_use" in sys.modules:
        from browser_use import Agent

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key or google_api_key.startswith("your_"):
        raise ValueError("GOOGLE_API_KEY missing.")
    if "ChatGoogleGenerativeAI" not in globals():
        raise ImportError("ChatGoogleGenerativeAI missing.")

    # Initialize the LLMs - Using gemini-2.0-flash for both
    main_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", google_api_key=google_api_key
    )
    planner_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", google_api_key=google_api_key
    )  # Keep defined for Agent init
    print(
        f"Using Main LLM: {getattr(main_llm, 'model', '?')}, Planner LLM configured as: {getattr(planner_llm, 'model', '?')}"
    )
    print("Note: Internal patch attempts to USE MAIN LLM during planning step.")

    # --- Define the NEW interactive form task ---
    form_filling_task = (
        "Navigate to the booking.com website and fill in the form with the following information: howeover you need think about the order as it may be different from this prompt"
        "Find the 'Hotels' search form section. Input 'London' as the destination city."
        "Attempt to select check-in and check-out dates approximately one month from today (Use May 1, 2025 for check-in and May 5, 2025 for check-out). "  # Using specific dates for clarity
        "Set the number of Adults to 2. "
        "Click the search button associated with the hotel form. "
        "Finally, report the main heading or title found on the search results page. Describe any difficulties encountered, especially with the date picker."
    )
    print(f"\nUsing New Task:\n{form_filling_task}\n")
    # --------------------------------------

    print("Initializing Agent...")
    # The Agent class itself now has the patched _run_planner method (if patch applied successfully)
    agent = Agent(
        task=form_filling_task,  # Use the new form filling task
        llm=main_llm,
        planner_llm=planner_llm,  # Still pass the planner object for settings/init checks
        use_vision=True,  # Vision might be helpful for complex forms
        use_vision_for_planner=False,  # This setting might be moot if main LLM is used by patch
        planner_interval=4,  # Keep planner active to ensure patch runs if needed
        save_conversation_path="logs/conversation",
    )

    async def run_agent():
        print("Running browser-use agent (with patched planner execution)...")
        # --- Set max_steps here ---
        max_steps_to_run = 150  # Keep increased max_steps
        print(f"Running agent for maximum {max_steps_to_run} steps...")
        history = await agent.run(max_steps=max_steps_to_run)
        # --------------------------
        print("\nAgent Run Completed.")
        steps_taken = (
            agent.state.n_steps - 1 if hasattr(agent, "state") else len(history.history)
        )
        print(f"Agent ran for {steps_taken} steps.")
        print("Visited URLs:", history.urls())
        final_result = history.final_result()
        print("Final result type:", type(final_result))
        if final_result is not None:
            print("Final result:", final_result)
        else:
            print("Final result: None")
        save_to_file(final_result, filename="agent_output.json")

    if "asyncio" not in sys.modules:
        print("ERROR: asyncio module not imported!")
        sys.exit(1)
    asyncio.run(run_agent())
    print("Browser-use test completed successfully!")

except ImportError as e:
    print(f"ImportError during browser-use test: {e}")
except ValueError as e:
    print(f"ValueError during browser-use test: {e}")
except Exception as e:
    print(f"Error running browser-use test: {e}")
    print(f"Detailed error for browser-use test:")
    print(f"{traceback.format_exc()}")
# === End of browser-use Test Section ===

# --- Restore Patches ---
# Restore Agent Patch
if (
    AGENT_CLASS_AVAILABLE
    and original_agent_run_planner
    and hasattr(Agent, target_method_name)
):
    try:
        setattr(Agent, target_method_name, original_agent_run_planner)
        print(f"\nRestored original Agent method: {target_method_name}")
    except Exception as e:
        print(f"\nFailed to restore Agent patch: {e}")

# Restore Playwright Patch
if original_playwright_launch:
    try:
        # from playwright._impl._browser_type import BrowserType # Already imported earlier
        BrowserType.launch = original_playwright_launch
        print("Restored original Playwright launch method.")
    except NameError:
        print("\nCould not restore Playwright launch method (BrowserType not defined).")
    except Exception as e:
        print(f"\nFailed to restore Playwright launch method: {e}")
# --- End Restore Patches ---
