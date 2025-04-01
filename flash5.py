"""Check browser installation and accessibility using patches.
   Includes:
   - Playwright launch patch.
   - Agent._run_planner patch (using main LLM).
   - Agent.step patch (attempting self-correction for errors).
   - Increased max_steps. Task: Research London Tech Conferences.
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
import logging # Import logging for patch logger
import time # Import time for patch timing

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
    from browser_use.telemetry.views import AgentStepTelemetryEvent # For telemetry in patch
    from browser_use.agent.views import AgentStepInfo, ActionResult, StepMetadata # For history/step patch
    from pydantic import ValidationError # To catch parsing errors
    from langchain_core.messages import HumanMessage # To add correction prompt


    print("Imported browser_use.Agent and patch dependencies.")
    AGENT_CLASS_AVAILABLE = True
except ImportError as e:
    print(f"WARN: Could not import browser_use.Agent or patch dependencies: {e}. Patches may not apply correctly.")
    AGENT_CLASS_AVAILABLE = False
# --- End Early Imports ---


# Store the original methods globally
original_playwright_launch = None
original_agent_run_planner = None
original_agent_step = None # To store original Agent.step

# === Start browser_use Agent Patch for _run_planner ===
if AGENT_CLASS_AVAILABLE:
    target_method_name_planner = "_run_planner"
    if hasattr(Agent, target_method_name_planner):
        original_agent_run_planner = getattr(Agent, target_method_name_planner)
        print(f"Stored original Agent method: {target_method_name_planner}")

        async def patched_run_planner(self, *args, **kwargs):
            """Patched version to use main LLM instead of planner LLM."""
            global original_agent_run_planner
            print(f"DEBUG Patch: Intercepting {target_method_name_planner}. FORCING MAIN LLM FOR PLANNING.")
            main_llm = getattr(self, 'llm', None)
            if not self.settings.planner_llm or not main_llm:
                print("DEBUG Patch: Planner LLM not configured or Main LLM missing, skipping planner.")
                if not original_agent_run_planner: raise RuntimeError("Original _run_planner not captured!")
                return await original_agent_run_planner(self, *args, **kwargs)

            print(f"DEBUG Patch: Attempting planning using MAIN LLM ({getattr(main_llm, 'model', '?')}).")
            try:
                page = await self.browser_context.get_current_page()
                standard_actions = self.controller.registry.get_prompt_description()
                page_actions = self.controller.registry.get_prompt_description(page)
                all_actions = standard_actions
                if page_actions: all_actions += '\n' + page_actions

                planner_messages = [
                    PlannerPrompt(all_actions).get_system_message(self.settings.is_planner_reasoning),
                    *self._message_manager.get_messages()[1:],
                ]

                if not self.settings.use_vision_for_planner and self.settings.use_vision:
                    last_state_message = planner_messages[-1]
                    new_msg = ''
                    content_to_process = last_state_message.content
                    if isinstance(content_to_process, list):
                        for msg_item in content_to_process:
                            if isinstance(msg_item, dict) and msg_item.get('type') == 'text':
                                new_msg += msg_item.get('text', '')
                    elif isinstance(content_to_process, str): new_msg = content_to_process
                    if new_msg != content_to_process: planner_messages[-1] = type(last_state_message)(content=new_msg)

                main_model_name = getattr(main_llm, 'model', 'Unknown')
                planner_messages = convert_input_messages(planner_messages, main_model_name)

                response = await main_llm.ainvoke(planner_messages)
                plan = str(response.content)

                if main_model_name and ('deepseek-r1' in main_model_name or 'deepseek-reasoner' in main_model_name):
                     if hasattr(self,'_remove_think_tags'): plan = self._remove_think_tags(plan)

                log_func = getattr(self, 'logger', None); info_func = getattr(log_func, 'info', print) if log_func else print; debug_func = getattr(log_func, 'debug', print) if log_func else print
                try:
                    plan_json = json.loads(plan); info_func(f'Planning Analysis (via main LLM in patch):\n{json.dumps(plan_json, indent=4)}')
                except json.JSONDecodeError: info_func(f'Planning Analysis (via main LLM in patch):\n{plan}')
                except Exception as e: debug_func(f'Error parsing planning analysis: {e}'); info_func(f'Plan: {plan}')
                return plan
            except Exception as e:
                log_func = getattr(self, 'logger', None); error_func = getattr(log_func, 'error', print) if log_func else print
                error_func(f'Error during patched planning execution: {str(e)}')
                raise RuntimeError(f'Patched planning failed: {e}') from e

        setattr(Agent, target_method_name_planner, patched_run_planner)
        print(f"Applied patch to Agent method: {target_method_name_planner}")
    else: print(f"WARN: Could not find method '{target_method_name_planner}' on Agent class to patch.")
# === End browser_use Agent Patch for _run_planner ===

# === Start browser_use Agent Patch for step (Self-Correction) ===
if AGENT_CLASS_AVAILABLE:
    target_method_name_step = "step"
    if hasattr(Agent, target_method_name_step):
        original_agent_step = getattr(Agent, target_method_name_step)
        print(f"Stored original Agent method: {target_method_name_step}")

        async def patched_step(self, step_info: Optional[AgentStepInfo] = None) -> None:
            """Patched step method with error recovery logic."""
            global original_agent_step
            logger = getattr(self, 'logger', logging.getLogger(__name__))
            logger.info(f'📍 Patched Step {self.state.n_steps}')
            state = None; model_output = None; result: list[ActionResult] = []; step_start_time = time.time(); tokens = 0; correction_triggered_this_step = False

            try: # Outer try
                state = await self.browser_context.get_state(); active_page = await self.browser_context.get_current_page()
                if self.settings.enable_memory and self.memory and self.state.n_steps % self.settings.memory_interval == 0: self.memory.create_procedural_memory(self.state.n_steps)
                await self._raise_if_stopped_or_paused(); await self._update_action_models_for_page(active_page)
                self._message_manager.add_state_message(state, self.state.last_result, step_info, self.settings.use_vision)
                if self.settings.planner_llm and self.state.n_steps % self.settings.planner_interval == 0:
                    plan = await self._run_planner(); # Uses planner patch
                    if plan: self._message_manager.add_plan(plan, position=-1)
                if step_info and step_info.is_last_step():
                    msg = 'Now comes your last step. Use only the "done" action now...'; logger.info('Last step finishing up')
                    self._message_manager._add_message_with_tokens(HumanMessage(content=msg))
                    if hasattr(self, 'DoneAgentOutput'): self.AgentOutput = self.DoneAgentOutput
                input_messages = self._message_manager.get_messages(); tokens = self._message_manager.state.history.current_tokens

                try: # Inner try for Action Gen/Exec
                    model_output = await self.get_next_action(input_messages)
                    await self._raise_if_stopped_or_paused()
                    if self.register_new_step_callback:
                        callback_func = self.register_new_step_callback
                        if inspect.iscoroutinefunction(callback_func): await callback_func(state, model_output, self.state.n_steps + 1)
                        else: callback_func(state, model_output, self.state.n_steps + 1)
                    if self.settings.save_conversation_path: pass # Simplify: Assume save_conversation handled elsewhere if needed
                    self._message_manager._remove_last_state_message(); await self._raise_if_stopped_or_paused()
                    self._message_manager.add_model_output(model_output);
                    result = await self.multi_act(model_output.action)
                    self.state.last_result = result
                    if result and result[-1].is_done: logger.info(f'📄 Result: {result[-1].extracted_content}')
                    self.state.consecutive_failures = 0
                except (ValueError, ValidationError, LLMException) as recoverable_error: # Self-Correction Path
                    correction_triggered_this_step = True; logger.warning(f"⚠️ Step {self.state.n_steps}: Action generation/execution failed: {type(recoverable_error).__name__}. Attempting self-correction.")
                    error_summary = f"{type(recoverable_error).__name__}: {str(recoverable_error)}"
                    error_msg_for_llm = (f"ERROR: The previous attempt failed with error: '{error_summary}'. Analyze this error and the current browser state. Propose a *different* action or sequence of actions to recover or achieve the original goal via an alternative method. Do not simply repeat the failed action.")
                    try: self._message_manager._remove_last_state_message()
                    except IndexError: pass
                    self._message_manager._add_message_with_tokens(HumanMessage(content=error_msg_for_llm))
                    try:
                         current_state_after_fail = await self.browser_context.get_state()
                         self._message_manager.add_state_message(current_state_after_fail, None, step_info, self.settings.use_vision)
                    except Exception as state_err: logger.error(f"Failed to get browser state after error: {state_err}"); self._message_manager._add_message_with_tokens(HumanMessage(content="Failed to retrieve browser state after error."))
                    logger.info("⏭️ Requesting corrective action from LLM..."); result = [ActionResult(error=f"Attempting correction for: {type(recoverable_error).__name__}", include_in_memory=False)]; self.state.last_result = result; model_output = None

            except InterruptedError: logger.info('Agent step paused/interrupted.'); self.state.last_result = [ActionResult(error='Agent paused/interrupted', include_in_memory=False)]; raise
            except asyncio.CancelledError: logger.info('Agent step cancelled.'); self.state.last_result = [ActionResult(error='Step cancelled', include_in_memory=False)]; raise InterruptedError('Step cancelled by user')
            except Exception as e: # Catch other unexpected errors
                logger.error(f"❌ Unexpected error during step {self.state.n_steps}: {e}"); logger.debug(f"Full traceback for unexpected step error:\n{traceback.format_exc()}")
                self.state.consecutive_failures += 1
                # Try using AgentError formatter if available
                error_msg = str(e)
                if 'AgentError' in globals(): error_msg = AgentError.format_error(e, include_trace=False)
                result = [ActionResult(error=f"Unexpected Step Error: {error_msg}", include_in_memory=True)]; self.state.last_result = result
            finally: # Ensure step count increments unless interrupted
                step_end_time = time.time()
                if not isinstance(locals().get('e', None), (InterruptedError, asyncio.CancelledError)): self.state.n_steps += 1
                # Basic Telemetry / History (simplified from original)
                actions_taken = [a.model_dump(exclude_unset=True) for a in model_output.action] if model_output else []
                step_error = [r.error for r in result if r and r.error] if result else []
                # TODO: Consider adding telemetry capture back if needed
                if state and result:
                     metadata = StepMetadata(step_number=self.state.n_steps, step_start_time=step_start_time, step_end_time=step_end_time, input_tokens=tokens)
                     if hasattr(self, '_make_history_item'): self._make_history_item(model_output, state, result, metadata)
                     else: logger.warning("Could not find _make_history_item method.")

        setattr(Agent, target_method_name_step, patched_step)
        print(f"Applied patch to Agent method: {target_method_name_step}")
    else: print(f"WARN: Could not find method '{target_method_name_step}' on Agent class to patch.")
# === End browser_use Agent Patch for step ===


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

# (Redundant checks kept for structure)
try:
    import browser_use
    try: print(f"\nbrowser-use version: {browser_use.__version__}")
    except AttributeError: print("\nbrowser-use version: unknown")
except ImportError as e: print(f"\nFailed to import browser_use: {e}")
try:
    import playwright
    print(f"playwright version: {getattr(playwright, '__version__', 'unknown')}")
except ImportError as e: print(f"Failed to import playwright: {e}")


# --- Apply Playwright monkey patch ONCE here ---
print("\nApplying Playwright monkey patch...")
chromium_path = os.environ.get('PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH')
if not chromium_path or not os.path.exists(chromium_path):
    print(f"WARNING: PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH not set or file not found: {chromium_path}")
    print("Playwright Patch might not work correctly.")
    chromium_path = None
try:
    if original_playwright_launch is None:
        original_playwright_launch = BrowserType.launch
        print("Stored original Playwright launch method.")
    def patched_playwright_launch(self, **kwargs):
        global original_playwright_launch; current_chromium_path = os.environ.get('PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH')
        print(f"Patched Playwright launch called, forcing executablePath={current_chromium_path} and headless=True")
        if current_chromium_path and os.path.exists(current_chromium_path): kwargs['executablePath'] = current_chromium_path
        else: print(f"WARNING: Cannot force executablePath, invalid path: {current_chromium_path}")
        kwargs['headless'] = True;
        if 'env' not in kwargs or kwargs['env'] is None: kwargs['env'] = {}
        if isinstance(kwargs['env'], dict): kwargs['env']['PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS'] = 'true'
        else: print(f"WARNING: Playwright launch 'env' not a dict.")
        if original_playwright_launch: return original_playwright_launch(self, **kwargs)
        else: raise RuntimeError("Original Playwright launch method not captured!")
    BrowserType.launch = patched_playwright_launch
    print("Playwright monkey patch applied successfully.")
except Exception as e: print(f"Failed to apply Playwright monkey patch: {e}"); sys.exit(1)
# --- End Playwright monkey patch ---

# Define save_to_file function
def save_to_file(content, filename="agent_output.json"):
    try:
        output_dir = Path(filename).parent
        if output_dir: output_dir.mkdir(parents=True, exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            try:
                if isinstance(content, (dict, list)): json.dump(content, f, indent=4, ensure_ascii=False)
                else: f.write(str(content))
            except (TypeError, OverflowError): f.write(str(content))
        print(f"Output successfully saved to {filename}")
    except Exception as e: print(f"Error saving output to file '{filename}': {e}")

# === Testing browser-use with ALL Patches ===
print("\nTesting browser-use with Patched Agent._run_planner and Agent.step:")
try:
    if 'Agent' not in globals() and 'browser_use' in sys.modules: from browser_use import Agent
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key or google_api_key.startswith("your_"): raise ValueError("GOOGLE_API_KEY missing.")
    if 'ChatGoogleGenerativeAI' not in globals(): raise ImportError("ChatGoogleGenerativeAI missing.")

    main_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
    planner_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
    print(f"Using Main LLM: {getattr(main_llm, 'model', '?')}, Planner LLM configured as: {getattr(planner_llm, 'model', '?')}")
    print("Note: _run_planner patch attempts to USE MAIN LLM during planning step.")
    print("Note: step patch attempts self-correction on certain errors.")

    # --- Define the NEW, Harder Research Task ---
    conference_research_task = (
        "Research upcoming technology conferences. Find exactly three distinct technology conferences scheduled to happen **in London** within the **next 3 months** (starting from today, April 1, 2025). "
        "For each of the three conferences, extract and list its name, specific dates (or month if specific dates aren't available), venue/location within London, and the main theme or topics covered. "
        "Ignore any purely online events or webinars. Present the final list of three conferences and their details clearly."
    )
    print(f"\nUsing New Harder Task:\n{conference_research_task}\n")
    # -----------------------------------------

    print("Initializing Agent...")
    agent = Agent(
        task=conference_research_task, # Use the new research task
        llm=main_llm,
        planner_llm=planner_llm,
        use_vision=True,
        use_vision_for_planner=False,
        planner_interval=4,
        save_conversation_path="logs/conversation"
    )

    async def run_agent():
        print("Running browser-use agent (with patched planner execution and self-correction)...")
        max_steps_to_run = 150 # Keep increased steps for potentially complex task
        print(f"Running agent for maximum {max_steps_to_run} steps...")
        history = await agent.run(max_steps=max_steps_to_run)
        print("\nAgent Run Completed.")
        steps_taken = agent.state.n_steps -1 if hasattr(agent,'state') else len(history.history)
        print(f"Agent ran for {steps_taken} steps.")
        print("Visited URLs:", history.urls())
        final_result = history.final_result()
        print("Final result type:", type(final_result))
        if final_result is not None: print("Final result:", final_result)
        else: print("Final result: None")
        save_to_file(final_result, filename="agent_output.json")

    if 'asyncio' not in sys.modules: print("ERROR: asyncio module not imported!"); sys.exit(1)
    asyncio.run(run_agent())
    print("Browser-use test completed successfully!")

except ImportError as e: print(f"ImportError during browser-use test: {e}")
except ValueError as e: print(f"ValueError during browser-use test: {e}")
except Exception as e:
    print(f"Error running browser-use test: {e}")
    print(f"Detailed error for browser-use test:")
    print(f"{traceback.format_exc()}")
# === End of browser-use Test Section ===

# --- Restore Patches ---
# Restore Agent Patches
if AGENT_CLASS_AVAILABLE:
    if original_agent_run_planner and hasattr(Agent, '_run_planner'):
        try: setattr(Agent, '_run_planner', original_agent_run_planner)
        except Exception as e: print(f"\nFailed to restore Agent patch (_run_planner): {e}")
        else: print(f"\nRestored original Agent method: _run_planner")
    if original_agent_step and hasattr(Agent, 'step'):
        try: setattr(Agent, 'step', original_agent_step)
        except Exception as e: print(f"\nFailed to restore Agent patch (step): {e}")
        else: print(f"Restored original Agent method: step")

# Restore Playwright Patch
if original_playwright_launch:
    try: BrowserType.launch = original_playwright_launch
    except Exception as e: print(f"\nFailed to restore Playwright launch method: {e}")
    else: print("Restored original Playwright launch method.")
# --- End Restore Patches ---