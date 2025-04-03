"""ATTEMPTING ADVANCED TASK (HIGH RISK / EXPERIMENTAL) - v3 Decorator Fixed
   Includes:
   - Playwright launch patch.
   - Agent._run_planner patch (using main LLM).
   - Agent.step patch (attempting self-correction, suggesting JS on action failure).
   - execute_javascript_action custom action (SECURITY RISK!).
   - Task: Mimic browser-use site, generate HTML/CSS/JS, attempt validation via online editor.
"""

import os
import sys
import asyncio
import traceback
from pathlib import Path
from dotenv import load_dotenv
import inspect
import functools
import json
import logging  # Import logging for patch logger
import time  # Import time for patch timing

# --- Early Imports ---
try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    print("Imported ChatGoogleGenerativeAI.")
except ImportError as e:
    print(f"Failed to import ChatGoogleGenerativeAI: {e}")
    sys.exit(1)

try:
    from playwright.async_api import Page  # Added Page for JS action type hint
    from playwright._impl._browser_type import BrowserType
    from playwright.sync_api import sync_playwright

    print("Imported Playwright components.")
except ImportError as e:
    print(f"Failed to import Playwright components: {e}")
    sys.exit(1)

try:
    # Imports needed for Agent class and patching Agent.step
    # Added Controller, Browser, BrowserConfig. Removed AgentError.
    from browser_use import Agent, Controller, Browser, BrowserConfig, ActionResult
    from browser_use.agent.views import AgentStepInfo
    from browser_use.agent.prompts import PlannerPrompt
    from browser_use.agent.message_manager.utils import (
        convert_input_messages,
    )  # Needed in planner patch
    from browser_use.exceptions import LLMException
    from browser_use.telemetry.views import (
        AgentStepTelemetryEvent,
    )  # For telemetry in patch
    from browser_use.agent.views import StepMetadata  # For history in patch
    from pydantic import ValidationError  # To catch parsing errors
    from langchain_core.messages import HumanMessage  # To add correction prompt

    print("Imported browser_use components and patch dependencies.")
    AGENT_CLASS_AVAILABLE = True
except ImportError as e:
    print(
        f"WARN: Could not import browser_use components or patch dependencies: {e}. Patches may not apply correctly."
    )
    AGENT_CLASS_AVAILABLE = False
# --- End Early Imports ---


# Setup logger for patches
patch_logger = logging.getLogger("AgentPatches")
patch_logger.setLevel(logging.DEBUG)  # Or INFO
if not logging.getLogger().handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)


# Store the original methods globally
original_playwright_launch = None
original_agent_run_planner = None
original_agent_step = None

# === Start browser_use Agent Patch for _run_planner ===
if AGENT_CLASS_AVAILABLE:
    target_method_name_planner = "_run_planner"
    if hasattr(Agent, target_method_name_planner):
        original_agent_run_planner = getattr(Agent, target_method_name_planner)
        patch_logger.info(f"Stored original Agent method: {target_method_name_planner}")

        async def patched_run_planner(self, *args, **kwargs):
            # (Keep identical patched _run_planner logic from previous version)
            global original_agent_run_planner
            patch_logger.debug(f"PATCH _run_planner: Intercepting. FORCING MAIN LLM.")
            main_llm = getattr(self, "llm", None)
            if not self.settings.planner_llm or not main_llm:
                patch_logger.debug(
                    "PATCH _run_planner: Planner/Main LLM missing, skipping."
                )
                return await original_agent_run_planner(self, *args, **kwargs)
            patch_logger.debug(
                f"PATCH _run_planner: Attempting planning via MAIN LLM ({getattr(main_llm, 'model', '?')})."
            )
            try:
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
                main_model_name = getattr(main_llm, "model", "Unknown")
                planner_messages = convert_input_messages(
                    planner_messages, main_model_name
                )
                response = await main_llm.ainvoke(planner_messages)
                plan = str(response.content)
                if main_model_name and ("deepseek" in main_model_name):
                    if hasattr(self, "_remove_think_tags"):
                        plan = self._remove_think_tags(plan)
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
                return plan
            except Exception as e:
                log_func = getattr(self, "logger", None)
                error_func = getattr(log_func, "error", print) if log_func else print
                error_func(f"Error during patched planning execution: {str(e)}")
                raise RuntimeError(f"Patched planning failed: {e}") from e

        setattr(Agent, target_method_name_planner, patched_run_planner)
        patch_logger.info(
            f"Applied patch to Agent method: {target_method_name_planner}"
        )
    else:
        patch_logger.warning(
            f"Could not find method '{target_method_name_planner}' on Agent class to patch."
        )
# === End browser_use Agent Patch for _run_planner ===

# === Start browser_use Agent Patch for step (Self-Correction - AgentError removed) ===
if AGENT_CLASS_AVAILABLE:
    target_method_name_step = "step"
    if hasattr(Agent, target_method_name_step):
        original_agent_step = getattr(Agent, target_method_name_step)
        patch_logger.info(f"Stored original Agent method: {target_method_name_step}")

        async def patched_step(self, step_info: Optional[AgentStepInfo] = None) -> None:
            """Patched step method with enhanced error recovery logic."""
            global original_agent_step
            logger = getattr(self, "logger", patch_logger)
            logger.info(f"📍 Patched Step {self.state.n_steps}")
            state = None
            model_output = None
            result: list[ActionResult] = []
            step_start_time = time.time()
            tokens = 0
            correction_triggered_this_step = False
            try:  # Outer try
                state = await self.browser_context.get_state()
                active_page = await self.browser_context.get_current_page()
                if (
                    self.settings.enable_memory
                    and self.memory
                    and self.state.n_steps % self.settings.memory_interval == 0
                ):
                    self.memory.create_procedural_memory(self.state.n_steps)
                await self._raise_if_stopped_or_paused()
                await self._update_action_models_for_page(active_page)
                self._message_manager.add_state_message(
                    state, self.state.last_result, step_info, self.settings.use_vision
                )
                if (
                    self.settings.planner_llm
                    and self.state.n_steps % self.settings.planner_interval == 0
                ):
                    plan = await self._run_planner()
                    if plan:
                        self._message_manager.add_plan(plan, position=-1)
                if step_info and step_info.is_last_step():
                    msg = 'Now comes your last step. Use only the "done" action now...'
                    logger.info("Last step finishing up")
                    self._message_manager._add_message_with_tokens(
                        HumanMessage(content=msg)
                    )
                    if hasattr(self, "DoneAgentOutput"):
                        self.AgentOutput = self.DoneAgentOutput
                input_messages = self._message_manager.get_messages()
                tokens = self._message_manager.state.history.current_tokens
                try:  # Inner try for Action Gen/Exec
                    model_output = await self.get_next_action(input_messages)
                    await self._raise_if_stopped_or_paused()
                    if self.register_new_step_callback:
                        pass  # Simplified
                    self._message_manager._remove_last_state_message()
                    await self._raise_if_stopped_or_paused()
                    self._message_manager.add_model_output(model_output)
                    intended_actions_str = json.dumps(
                        [a.model_dump() for a in model_output.action], indent=2
                    )
                    result = await self.multi_act(model_output.action)
                    self.state.last_result = result
                    if result and result[-1].is_done:
                        logger.info(f"📄 Result: {result[-1].extracted_content}")
                    self.state.consecutive_failures = 0
                except (
                    ValueError,
                    ValidationError,
                    LLMException,
                ) as recoverable_error:  # Self-Correction for Parsing/LLM Errors
                    correction_triggered_this_step = True
                    logger.warning(
                        f"⚠️ Step {self.state.n_steps}: LLM response/parsing failed: {type(recoverable_error).__name__}. Attempting self-correction."
                    )
                    error_summary = (
                        f"{type(recoverable_error).__name__}: {str(recoverable_error)}"
                    )
                    error_msg_for_llm = f"ERROR: My previous response was invalid or caused an error: '{error_summary}'. Review the error and the current browser state. Generate a *valid* action sequence to achieve the original goal."
                    try:
                        self._message_manager._remove_last_state_message()
                    except IndexError:
                        pass
                    self._message_manager._add_message_with_tokens(
                        HumanMessage(content=error_msg_for_llm)
                    )
                    try:
                        current_state_after_fail = (
                            await self.browser_context.get_state()
                        )
                        self._message_manager.add_state_message(
                            current_state_after_fail,
                            None,
                            step_info,
                            self.settings.use_vision,
                        )
                    except Exception as state_err:
                        logger.error(f"Failed to get state after error: {state_err}")
                        self._message_manager._add_message_with_tokens(
                            HumanMessage(
                                content="Failed to retrieve browser state after error."
                            )
                        )
                    logger.info(
                        "⏭️ Requesting corrective action from LLM (parsing/LLM error)..."
                    )
                    result = [
                        ActionResult(
                            error=f"Attempting correction for: {type(recoverable_error).__name__}",
                            include_in_memory=False,
                        )
                    ]
                    self.state.last_result = result
                    model_output = None
                except (
                    Exception
                ) as action_error:  # Self-Correction for Action Execution Errors
                    correction_triggered_this_step = True
                    failed_action_name = "unknown"
                    if model_output and model_output.action:
                        failed_action_name = next(
                            iter(model_output.action[0].model_dump(exclude_unset=True)),
                            "unknown",
                        )
                    logger.warning(
                        f"⚠️ Step {self.state.n_steps}: Action execution failed ('{failed_action_name}'): {type(action_error).__name__}. Attempting self-correction."
                    )
                    error_summary = (
                        f"{type(action_error).__name__}: {str(action_error)}"
                    )
                    js_suggestion = ""
                    if (
                        "input_text" in failed_action_name
                        or "element" in str(action_error).lower()
                    ):
                        js_suggestion = " If standard input/click failed, consider using the 'execute_javascript_action' action with appropriate JS code (e.g., document.querySelector or getElementById) to directly manipulate the element as a potential workaround."
                    error_msg_for_llm = f"ERROR: The previous action attempt ('{failed_action_name}') failed during execution: '{error_summary}'. Analyze this error and the current browser state. Propose a *different* action or sequence of actions to recover.{js_suggestion}"
                    try:
                        self._message_manager._remove_last_state_message()
                    except IndexError:
                        pass
                    self._message_manager._add_message_with_tokens(
                        HumanMessage(content=error_msg_for_llm)
                    )
                    try:
                        current_state_after_fail = (
                            await self.browser_context.get_state()
                        )
                        self._message_manager.add_state_message(
                            current_state_after_fail,
                            None,
                            step_info,
                            self.settings.use_vision,
                        )
                    except Exception as state_err:
                        logger.error(f"Failed to get state after error: {state_err}")
                        self._message_manager._add_message_with_tokens(
                            HumanMessage(
                                content="Failed to retrieve browser state after error."
                            )
                        )
                    logger.info(
                        "⏭️ Requesting corrective action from LLM (action execution error)..."
                    )
                    result = [
                        ActionResult(
                            error=f"Attempting correction for: {type(action_error).__name__}",
                            include_in_memory=False,
                        )
                    ]
                    self.state.last_result = result
                    model_output = None

            except InterruptedError:
                logger.info("Agent step paused/interrupted.")
                self.state.last_result = [
                    ActionResult(
                        error="Agent paused/interrupted", include_in_memory=False
                    )
                ]
                raise
            except asyncio.CancelledError:
                logger.info("Agent step cancelled.")
                self.state.last_result = [
                    ActionResult(error="Step cancelled", include_in_memory=False)
                ]
                raise InterruptedError("Step cancelled by user")
            except Exception as e:  # Catch unexpected errors in outer try
                logger.error(
                    f"❌ Unexpected error during step {self.state.n_steps}: {e}"
                )
                logger.debug(
                    f"Full traceback for unexpected step error:\n{traceback.format_exc()}"
                )
                self.state.consecutive_failures += 1
                error_msg = str(e)
                # Removed AgentError handling
                result = [
                    ActionResult(
                        error=f"Unexpected Step Error: {error_msg}",
                        include_in_memory=True,
                    )
                ]
                self.state.last_result = result
            finally:
                step_end_time = time.time()
                if not isinstance(
                    locals().get("e", None), (InterruptedError, asyncio.CancelledError)
                ):
                    self.state.n_steps += 1
                actions_taken = (
                    [a.model_dump(exclude_unset=True) for a in model_output.action]
                    if model_output
                    else []
                )
                step_error = (
                    [r.error for r in result if r and r.error] if result else []
                )
                if hasattr(self, "telemetry") and hasattr(self.telemetry, "capture"):
                    self.telemetry.capture(
                        AgentStepTelemetryEvent(
                            agent_id=self.state.agent_id,
                            step=self.state.n_steps,
                            actions=actions_taken,
                            consecutive_failures=self.state.consecutive_failures,
                            step_error=step_error,
                        )
                    )
                if state and result:
                    metadata = StepMetadata(
                        step_number=self.state.n_steps,
                        step_start_time=step_start_time,
                        step_end_time=step_end_time,
                        input_tokens=tokens,
                    )
                    if hasattr(self, "_make_history_item"):
                        self._make_history_item(model_output, state, result, metadata)

        setattr(Agent, target_method_name_step, patched_step)
        print(f"Applied patch to Agent method: {target_method_name_step}")
    else:
        print(
            f"WARN: Could not find method '{target_method_name_step}' on Agent class to patch."
        )
# === End browser_use Agent Patch for step ===

# --- Define Custom Controller with JavaScript Execution ---
# 🚨 SECURITY WARNING: Executing LLM-generated JS is highly risky! 🚨
controller = Controller()  # Start with default controller


@controller.action(
    # Description is the FIRST positional argument
    "Executes a given snippet of Javascript code directly on the current page and returns the result. Use ONLY when standard actions (click, input) fail or are insufficient for interacting with complex elements. Verify element selectors carefully. Be extremely cautious, as this can impact page state unexpectedly."
    # No 'name=' argument needed here
)
async def execute_javascript_action(code: str, browser: Browser) -> ActionResult:
    """Executes Javascript code on the current page."""
    result_str = "JavaScript execution attempted."
    error_str = None
    extracted_content = None
    patch_logger.warning(
        f"Attempting to execute potentially risky JS: {code[:100]}..."
    )  # Log attempt
    try:
        page: Page = await browser.get_current_page()
        js_result = await page.evaluate(code)
        result_str = f"JavaScript executed successfully."
        if js_result is not None:
            try:
                extracted_content = json.dumps(js_result)
                result_str += f" Result (JSON): {extracted_content}"
            except TypeError:
                extracted_content = str(js_result)
                result_str += f" Result (string): {extracted_content}"
        patch_logger.info(f"JS Execution successful. Result: {extracted_content}")
    except Exception as e:
        error_str = f"Error executing JavaScript: {type(e).__name__}: {e}"
        patch_logger.error(error_str)
        return ActionResult(
            error=error_str, extracted_content=None, result_details={"js_code": code}
        )
    return ActionResult(
        extracted_content=result_str,
        result_details={"js_code": code, "js_output": extracted_content},
    )


print(
    f"Custom controller created with actions: {list(controller.registry.registry.actions.keys())}"
)
# --- End Custom Controller Definition ---


# Load environment variables from .env file
load_dotenv()
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Working directory: {os.getcwd()}")
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
try:
    import browser_use

    print(f"\nbrowser-use version: {getattr(browser_use, '__version__', 'unknown')}")
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
    chromium_path = None
try:
    if original_playwright_launch is None:
        original_playwright_launch = BrowserType.launch
        print("Stored original Playwright launch method.")

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


# === Testing browser-use with ALL Patches & Custom Controller ===
print("\nTesting browser-use with Patched Agent Methods and JS Execution Action:")
try:
    if "Agent" not in globals() and "browser_use" in sys.modules:
        from browser_use import Agent
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key or google_api_key.startswith("your_"):
        raise ValueError("GOOGLE_API_KEY missing.")
    if "ChatGoogleGenerativeAI" not in globals():
        raise ImportError("ChatGoogleGenerativeAI missing.")

    main_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", google_api_key=google_api_key
    )
    planner_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", google_api_key=google_api_key
    )
    print(
        f"Using Main LLM: {getattr(main_llm, 'model', '?')}, Planner LLM configured as: {getattr(planner_llm, 'model', '?')}"
    )
    print("Note: _run_planner patch attempts to USE MAIN LLM during planning step.")
    print(
        "Note: step patch attempts self-correction on certain errors, potentially suggesting JS."
    )

    # --- Define the Online Editor Task ---
    online_editor_task = (
        "1. Navigate to the main page of the browser-use documentation site (docs.browser-use.com). "
        "2. Extract the main headline and the first descriptive paragraph. "
        "3. Generate simple HTML structure (`<html>...`) using the extracted content. "
        "4. Generate basic CSS to style H1 (blue) and P (14px font). "
        "5. Generate a simple Javascript snippet (alert 'Loaded!'). "
        "6. Navigate to the online HTML editor (https://html-css-js.com/). Accept cookies if necessary. "
        "7. **Attempt** to input the generated HTML, CSS, and Javascript into the respective input areas (look for elements like textareas or specific editor divs). Use standard actions first. If input fails, consider using execute_javascript_action with e.g., document.getElementById('htmlCodeId').value = '...'. "
        "8. **Attempt** to click the 'Run' button. "
        "9. **Attempt** to extract text content from the output/result frame. "
        "10. Use 'done', report the generated code, extracted output, and difficulties encountered, especially with steps 7-9."
    )
    print(f"\nUsing Online Editor Task:\n{online_editor_task}\n")
    # -----------------------------------

    print("Initializing Agent (with custom controller)...")
    agent = Agent(
        task=online_editor_task,
        llm=main_llm,
        planner_llm=planner_llm,
        controller=controller,  # Pass the controller with the JS action
        use_vision=True,
        use_vision_for_planner=False,
        planner_interval=4,
        save_conversation_path="logs/conversation",
    )

    async def run_agent():
        print(
            "Running browser-use agent (with JS action and enhanced self-correction)..."
        )
        max_steps_to_run = 150
        print(f"Running agent for maximum {max_steps_to_run} steps...")
        history = await agent.run(max_steps=max_steps_to_run)
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
if AGENT_CLASS_AVAILABLE:  # Check if class was available before trying to restore
    target_method_name_planner = "_run_planner"
    target_method_name_step = "step"
    if original_agent_run_planner and hasattr(Agent, target_method_name_planner):
        try:
            setattr(Agent, target_method_name_planner, original_agent_run_planner)
        except Exception as e:
            print(
                f"\nFailed to restore Agent patch ({target_method_name_planner}): {e}"
            )
        else:
            print(f"\nRestored original Agent method: {target_method_name_planner}")
    if original_agent_step and hasattr(Agent, target_method_name_step):
        try:
            setattr(Agent, target_method_name_step, original_agent_step)
        except Exception as e:
            print(f"\nFailed to restore Agent patch ({target_method_name_step}): {e}")
        else:
            print(f"Restored original Agent method: {target_method_name_step}")
if original_playwright_launch:
    try:
        BrowserType.launch = original_playwright_launch
    except Exception as e:
        print(f"\nFailed to restore Playwright launch method: {e}")
    else:
        print("Restored original Playwright launch method.")
# --- End Restore Patches ---
