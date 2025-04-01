"""ATTEMPTING ADVANCED TASK (HIGH RISK / EXPERIMENTAL) - v19 Final Variable Fix
   Includes:
   - Playwright launch patch.
   - Agent._run_planner patch (FORCING main LLM for planning).
   - Agent.step patch (self-correction AND calls planner on error).
   - execute_javascript_action custom action (SECURITY RISK!).
   - extract_iframe_content custom action.
   - Task: Use online editor, attempt analysis.
   - Interactive chat loop after agent run.
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
import logging
import time
from typing import Optional, List, Dict, Any, Callable, Union, TypeVar

# --- Early Imports ---
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    print("Imported ChatGoogleGenerativeAI and Message types.")
except ImportError as e: print(f"Failed to import ChatGoogleGenerativeAI/Messages: {e}"); sys.exit(1)

try:
    from playwright.async_api import Page
    from playwright._impl._browser_type import BrowserType
    from playwright.sync_api import sync_playwright
    print("Imported Playwright components.")
except ImportError as e: print(f"Failed to import Playwright components: {e}"); sys.exit(1)

try:
    from browser_use import Agent, Controller, Browser, BrowserConfig, ActionResult
    from browser_use.agent.views import AgentStepInfo
    from browser_use.agent.prompts import PlannerPrompt
    from browser_use.agent.message_manager.utils import convert_input_messages
    # from browser_use.exceptions import LLMException # Keep removed
    from browser_use.telemetry.views import AgentStepTelemetryEvent
    from browser_use.agent.views import StepMetadata
    from pydantic import ValidationError, BaseModel, Field
    from browser_use.agent.memory.service import Memory, MemorySettings # Needed for init patch
    print("Imported browser_use components and patch dependencies.")
    AGENT_CLASS_AVAILABLE = True
except ImportError as e: print(f"WARN: Could not import browser_use components or patch dependencies: {e}. Patches may not apply correctly."); AGENT_CLASS_AVAILABLE = False
# --- End Early Imports ---

# Setup logger for patches
patch_logger = logging.getLogger("AgentPatches"); patch_logger.setLevel(logging.INFO)
if not logging.getLogger().handlers: handler = logging.StreamHandler(sys.stdout); formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'); handler.setFormatter(formatter); logging.getLogger().addHandler(handler); logging.getLogger().setLevel(logging.INFO)

# Store original methods
original_playwright_launch = None; original_agent_run_planner = None; original_agent_step = None
original_agent_init = None

# === Start browser_use Agent Patch for __init__ (Fix Memory Init) ===
if AGENT_CLASS_AVAILABLE:
    target_method_name_init = "__init__"
    if hasattr(Agent, target_method_name_init):
        original_agent_init = getattr(Agent, target_method_name_init); patch_logger.info(f"Stored original Agent method: {target_method_name_init}")
        init_sig = inspect.signature(original_agent_init)
        @functools.wraps(original_agent_init)
        def patched_init(self, *args, **kwargs):
            global original_agent_init; patch_logger.debug("PATCH __init__: Running original init...")
            original_agent_init(self, *args, **kwargs); patch_logger.debug("PATCH __init__: Original init finished. Applying memory fix.")
            bound_args = init_sig.bind(self, *args, **kwargs); bound_args.apply_defaults()
            enable_memory = bound_args.arguments.get('enable_memory'); memory_interval = bound_args.arguments.get('memory_interval'); memory_config = bound_args.arguments.get('memory_config')
            if enable_memory:
                patch_logger.debug(f"PATCH __init__: Initializing Memory object.")
                memory_settings = MemorySettings(agent_id=self.state.agent_id, interval=memory_interval, config=memory_config)
                if not hasattr(self, '_message_manager'): patch_logger.error("PATCH __init__ Error: self._message_manager missing."); return
                if not hasattr(self, 'llm'): patch_logger.error("PATCH __init__ Error: self.llm missing."); return
                self.memory = Memory(message_manager=self._message_manager, llm=self.llm, settings=memory_settings)
                patch_logger.debug(f"PATCH __init__: self.memory initialized.")
            else: patch_logger.debug(f"PATCH __init__: Setting self.memory to None."); self.memory = None
            patch_logger.info("PATCH __init__: Memory initialization fix applied.")
        setattr(Agent, target_method_name_init, patched_init); patch_logger.info(f"Applied patch to Agent method: {target_method_name_init}")
    else: patch_logger.warning(f"Could not find method '{target_method_name_init}' on Agent class to patch.")
# === End browser_use Agent Patch for __init__ ===


# === Start browser_use Agent Patch for _run_planner (REINSTATED) ===
if AGENT_CLASS_AVAILABLE:
    target_method_name_planner = "_run_planner"
    if hasattr(Agent, target_method_name_planner):
        original_agent_run_planner = getattr(Agent, target_method_name_planner); patch_logger.info(f"Stored original Agent method: {target_method_name_planner}")
        async def patched_run_planner(self, *args, **kwargs):
            global original_agent_run_planner; patch_logger.debug(f"PATCH _run_planner: Intercepting. FORCING MAIN LLM.")
            main_llm = getattr(self, 'llm', None)
            if not self.settings.planner_llm or not main_llm: patch_logger.debug("PATCH _run_planner: Planner/Main LLM missing, skipping."); return await original_agent_run_planner(self, *args, **kwargs)
            patch_logger.debug(f"PATCH _run_planner: Attempting planning via MAIN LLM ({getattr(main_llm, 'model', '?')}).")
            try:
                page = await self.browser_context.get_current_page(); standard_actions = self.controller.registry.get_prompt_description(); page_actions = self.controller.registry.get_prompt_description(page); all_actions = standard_actions
                if page_actions: all_actions += '\n' + page_actions
                planner_messages = [PlannerPrompt(all_actions).get_system_message(self.settings.is_planner_reasoning), *self._message_manager.get_messages()[1:]]
                if not self.settings.use_vision_for_planner and self.settings.use_vision:
                    last_state_message = planner_messages[-1]; new_msg = ''; content_to_process = last_state_message.content
                    if isinstance(content_to_process, list):
                        for msg_item in content_to_process:
                            if isinstance(msg_item, dict) and msg_item.get('type') == 'text': new_msg += msg_item.get('text', '')
                    elif isinstance(content_to_process, str): new_msg = content_to_process
                    if new_msg != content_to_process: planner_messages[-1] = type(last_state_message)(content=new_msg)
                main_model_name = getattr(main_llm, 'model', 'Unknown'); planner_messages = convert_input_messages(planner_messages, main_model_name)
                response = await main_llm.ainvoke(planner_messages); plan = str(response.content)
                if main_model_name and ('deepseek' in main_model_name):
                     if hasattr(self,'_remove_think_tags'): plan = self._remove_think_tags(plan)
                log_func = getattr(self, 'logger', None); info_func = getattr(log_func, 'info', print) if log_func else print; debug_func = getattr(log_func, 'debug', print) if log_func else print
                try: plan_json = json.loads(plan); info_func(f'Planning Analysis (via main LLM in patch):\n{json.dumps(plan_json, indent=4)}')
                except json.JSONDecodeError: info_func(f'Planning Analysis (via main LLM in patch):\n{plan}')
                except Exception as e: debug_func(f'Error parsing planning analysis: {e}'); info_func(f'Plan: {plan}')
                return plan
            except Exception as e: log_func = getattr(self, 'logger', None); error_func = getattr(log_func, 'error', print) if log_func else print; error_func(f'Error during patched planning execution: {str(e)}'); raise RuntimeError(f'Patched planning failed: {e}') from e
        setattr(Agent, target_method_name_planner, patched_run_planner); patch_logger.info(f"Applied patch to Agent method: {target_method_name_planner}")
    else: patch_logger.warning(f"Could not find method '{target_method_name_planner}' on Agent class to patch.")
# === End browser_use Agent Patch for _run_planner ===

# === Start browser_use Agent Patch for step (Self-Correction + Planner-on-Error + Method Call Fix) ===
if AGENT_CLASS_AVAILABLE:
    target_method_name_step = "step"
    if hasattr(Agent, target_method_name_step):
        original_agent_step = getattr(Agent, target_method_name_step); patch_logger.info(f"Stored original Agent method: {target_method_name_step}")
        async def patched_step(self, step_info: Optional[AgentStepInfo] = None) -> None:
            """Patched step method with corrected memory check, explicit method call & error recovery."""
            global original_agent_step; logger = getattr(self, 'logger', patch_logger); logger.info(f'📍 Patched Step {self.state.n_steps}')
            state = None; model_output = None; result: list[ActionResult] = []; step_start_time = time.time(); tokens = 0; correction_triggered_this_step = False; run_planner_now = False
            try: # Outer try
                state = await self.browser_context.get_state(); active_page = await self.browser_context.get_current_page()
                if hasattr(self,'memory') and self.memory and hasattr(self.memory, 'settings') and hasattr(self.memory.settings, 'interval') and self.state.n_steps % self.memory.settings.interval == 0:
                     if hasattr(self.memory, 'create_procedural_memory') and callable(getattr(self.memory, 'create_procedural_memory')): self.memory.create_procedural_memory(self.state.n_steps)
                await self._raise_if_stopped_or_paused()
                # Explicitly call via the class, passing the instance 'self'
                if hasattr(Agent, '_update_action_models_for_page'): await Agent._update_action_models_for_page(self, active_page)
                else: logger.warning("'_update_action_models_for_page' not found on Agent, skipping.")
                self._message_manager.add_state_message(state, self.state.last_result, step_info, self.settings.use_vision)
                if self.settings.planner_llm and self.state.n_steps % self.settings.planner_interval == 0:
                     run_planner_now = True; logger.info(f"Triggering planner based on interval ({self.state.n_steps} % {self.settings.planner_interval} == 0)")
                     if hasattr(self, '_run_planner') and callable(getattr(self, '_run_planner')): plan = await self._run_planner(); # Calls PATCHED
                     if plan: self._message_manager.add_plan(plan, position=-1)
                if step_info and step_info.is_last_step(): msg = 'Now comes your last step...'; logger.info('Last step finishing up'); self._message_manager._add_message_with_tokens(HumanMessage(content=msg)); # Simplified
                input_messages = self._message_manager.get_messages(); tokens = self._message_manager.state.history.current_tokens
                try: # Inner try for Action Gen/Exec
                    model_output = await self.get_next_action(input_messages); await self._raise_if_stopped_or_paused()
                    if self.register_new_step_callback: pass # Simplified
                    self._message_manager._remove_last_state_message(); await self._raise_if_stopped_or_paused()
                    self._message_manager.add_model_output(model_output);
                    intended_actions_str = json.dumps([a.model_dump() for a in model_output.action], indent=2)
                    result = await self.multi_act(model_output.action)
                    self.state.last_result = result;
                    if result and result[-1].is_done: logger.info(f'📄 Result: {result[-1].extracted_content}')
                    self.state.consecutive_failures = 0
                except (ValueError, ValidationError) as recoverable_error: # Catch Parsing errors
                    correction_triggered_this_step = True; logger.warning(f"⚠️ Step {self.state.n_steps}: Parsing/LLM error: {type(recoverable_error).__name__}. Correcting.")
                    error_summary = f"{type(recoverable_error).__name__}: {str(recoverable_error)}"
                    error_msg_for_llm = (f"ERROR: Previous response invalid: '{error_summary}'. Review state & provide valid action.")
                    try: self._message_manager._remove_last_state_message()
                    except IndexError: pass
                    self._message_manager._add_message_with_tokens(HumanMessage(content=error_msg_for_llm))
                    try: current_state_after_fail = await self.browser_context.get_state(); self._message_manager.add_state_message(current_state_after_fail, None, step_info, self.settings.use_vision)
                    except Exception as state_err: logger.error(f"Failed get state post-error: {state_err}"); self._message_manager._add_message_with_tokens(HumanMessage(content="Failed state retrieval."))
                    if self.settings.planner_llm and not run_planner_now: # Call planner on error
                        logger.info(f"Triggering planner due to error: {type(recoverable_error).__name__}")
                        if hasattr(self, '_run_planner') and callable(getattr(self, '_run_planner')): plan = await self._run_planner(); # Calls PATCHED
                        if plan: self._message_manager.add_plan(plan, position=-1)
                    logger.info("⏭️ Requesting corrective action (parsing/LLM error)..."); result = [ActionResult(error=f"Correcting: {type(recoverable_error).__name__}", include_in_memory=False)]; self.state.last_result = result; model_output = None
                except Exception as action_error: # Catch Action execution errors
                    correction_triggered_this_step = True; failed_action_name = "unknown";
                    if model_output and model_output.action: failed_action_name = next(iter(model_output.action[0].model_dump(exclude_unset=True)), "unknown")
                    logger.warning(f"⚠️ Step {self.state.n_steps}: Action failed ('{failed_action_name}'): {type(action_error).__name__}. Correcting.")
                    error_summary = f"{type(action_error).__name__}: {str(action_error)}"; js_suggestion = "";
                    if "input_text" in failed_action_name or "click_element" in failed_action_name or "element" in str(action_error).lower(): js_suggestion = " If standard input/click failed, consider using 'execute_javascript_action' with correct JS (e.g., querySelector/getElementById + .value= or .click()) as workaround."
                    error_msg_for_llm = (f"ERROR: Action '{failed_action_name}' failed: '{error_summary}'. Analyze state & propose *different* recovery action.{js_suggestion}")
                    try: self._message_manager._remove_last_state_message()
                    except IndexError: pass
                    self._message_manager._add_message_with_tokens(HumanMessage(content=error_msg_for_llm))
                    try: current_state_after_fail = await self.browser_context.get_state(); self._message_manager.add_state_message(current_state_after_fail, None, step_info, self.settings.use_vision)
                    except Exception as state_err: logger.error(f"Failed get state post-error: {state_err}"); self._message_manager._add_message_with_tokens(HumanMessage(content="Failed state retrieval."))
                    if self.settings.planner_llm and not run_planner_now: # Call planner on error
                        logger.info(f"Triggering planner due to error: {type(action_error).__name__}")
                        if hasattr(self, '_run_planner') and callable(getattr(self, '_run_planner')): plan = await self._run_planner(); # Calls PATCHED
                        if plan: self._message_manager.add_plan(plan, position=-1)
                    logger.info("⏭️ Requesting corrective action (action execution error)..."); result = [ActionResult(error=f"Correcting: {type(action_error).__name__}", include_in_memory=False)]; self.state.last_result = result; model_output = None

            except InterruptedError: logger.info('Agent step paused/interrupted.'); self.state.last_result = [ActionResult(error='Agent paused/interrupted')]; raise
            except asyncio.CancelledError: logger.info('Agent step cancelled.'); self.state.last_result = [ActionResult(error='Step cancelled')]; raise InterruptedError('Step cancelled')
            except Exception as e: # Catch unexpected errors in outer try
                logger.error(f"❌ Unexpected outer error step {self.state.n_steps}: {e}"); logger.debug(f"Traceback:\n{traceback.format_exc()}")
                self.state.consecutive_failures += 1; error_msg = str(e)
                # Removed AgentError check
                result = [ActionResult(error=f"Unexpected Step Error: {error_msg}", include_in_memory=True)]; self.state.last_result = result
            finally: # Ensure step count increments unless interrupted
                step_end_time = time.time();
                if not isinstance(locals().get('e', None), (InterruptedError, asyncio.CancelledError)): self.state.n_steps += 1
                actions_taken = [a.model_dump(exclude_unset=True) for a in model_output.action] if model_output else []; step_error = [r.error for r in result if r and r.error] if result else []
                if hasattr(self, 'telemetry') and hasattr(self.telemetry, 'capture'): self.telemetry.capture( AgentStepTelemetryEvent( agent_id=self.state.agent_id, step=self.state.n_steps, actions=actions_taken, consecutive_failures=self.state.consecutive_failures, step_error=step_error ) )
                if state and result:
                     metadata = StepMetadata(step_number=self.state.n_steps, step_start_time=step_start_time, step_end_time=step_end_time, input_tokens=tokens)
                     if hasattr(self, '_make_history_item'): self._make_history_item(model_output, state, result, metadata)

        setattr(Agent, target_method_name_step, patched_step); print(f"Applied patch to Agent method: {target_method_name_step}")
    else: print(f"WARN: Could not find method '{target_method_name_step}' on Agent class to patch.")
# === End browser_use Agent Patch for step ===

# --- Define Custom Controller with JavaScript Execution & IFrame Extraction ---
# (Keep identical custom controller logic)
controller = Controller()
@controller.action('Executes JS code. Use ONLY when standard actions fail. Be cautious.')
async def execute_javascript_action(code: str, browser: Browser) -> ActionResult:
    result_str = "JavaScript execution attempted."; error_str = None; extracted_content = None; patch_logger.warning(f"Attempting JS: {code[:100]}...")
    try:
        page: Page = await browser.get_current_page(); js_result = await page.evaluate(code); result_str = f"JavaScript executed successfully."
        if js_result is not None:
            try: extracted_content = json.dumps(js_result); result_str += f" Result (JSON): {extracted_content}"
            except TypeError: extracted_content = str(js_result); result_str += f" Result (string): {extracted_content}"
        patch_logger.info(f"JS Execution successful. Result: {extracted_content}")
    except Exception as e: error_str = f"Error executing JavaScript: {type(e).__name__}: {e}"; patch_logger.error(error_str); return ActionResult(error=error_str, result_details={"js_code": code})
    return ActionResult(extracted_content=result_str, result_details={"js_code": code, "js_output": extracted_content})
class IframeContentParams(BaseModel):
    iframe_selector: str = Field(..., description="CSS selector for the iframe element.")
    extract_text: bool = Field(default=True, description="True for text content, False for inner HTML.")
@controller.action('Extracts content from within a specified iframe.', param_model=IframeContentParams)
async def extract_iframe_content(params: IframeContentParams, browser: Browser) -> ActionResult:
    patch_logger.info(f"Attempting iframe extract: {params.iframe_selector}")
    try:
        page: Page = await browser.get_current_page(); frame_locator = page.frame_locator(params.iframe_selector)
        try: await frame_locator.locator(':root').wait_for(state='visible', timeout=5000)
        except Exception: return ActionResult(error=f"IFrame '{params.iframe_selector}' not found/visible.")
        content = ""; mode = "text" if params.extract_text else "html"
        if params.extract_text: content = await frame_locator.locator('body').inner_text(timeout=5000)
        else: content = await frame_locator.locator(':root').inner_html(timeout=5000)
        result_str = f"Successfully extracted {mode} content from iframe '{params.iframe_selector}'."; patch_logger.info(f"IFrame Extraction success (length: {len(content)}).")
        max_len = 2000;
        if len(content) > max_len: content = content[:max_len] + f"... (truncated)"
        return ActionResult(extracted_content=content, result_details={"selector": params.iframe_selector, "mode": mode})
    except Exception as e: error_str = f"Error extracting from iframe '{params.iframe_selector}': {type(e).__name__}: {e}"; patch_logger.error(error_str); return ActionResult(error=error_str)
print(f"Custom controller created with actions: {list(controller.registry.registry.actions.keys())}")
# --- End Custom Controller Definition ---

# Load environment variables from .env file
load_dotenv()
print(f"Python version: {sys.version}"); print(f"Python executable: {sys.executable}"); print(f"Working directory: {os.getcwd()}")
print("\nEnvironment variables:")
for key in ['LD_LIBRARY_PATH', 'PLAYWRIGHT_BROWSERS_PATH', 'BROWSER_USE_BROWSER_TYPE','BROWSER_USE_HEADLESS', 'PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS','PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH']: print(f"{key}: {os.environ.get(key, 'Not set')}")
try: import browser_use; print(f"\nbrowser-use version: {getattr(browser_use, '__version__', 'unknown')}")
except ImportError as e: print(f"\nFailed to import browser_use: {e}")
try: import playwright; print(f"playwright version: {getattr(playwright, '__version__', 'unknown')}")
except ImportError as e: print(f"Failed to import playwright: {e}")

# --- Apply Playwright monkey patch ONCE here ---
print("\nApplying Playwright monkey patch...")
chromium_path = os.environ.get('PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH')
if not chromium_path or not os.path.exists(chromium_path): print(f"WARNING: PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH not set or file not found: {chromium_path}"); chromium_path = None
try:
    if original_playwright_launch is None: original_playwright_launch = BrowserType.launch; print("Stored original Playwright launch method.")
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
    BrowserType.launch = patched_playwright_launch; print("Playwright monkey patch applied successfully.")
except Exception as e: print(f"Failed to apply Playwright monkey patch: {e}"); sys.exit(1)
# --- End Playwright monkey patch ---

# Define save_to_file function
def save_to_file(content, filename="agent_output.json"):
    # (Keep identical save_to_file logic)
    try:
        output_dir = Path(filename).parent;
        if output_dir: output_dir.mkdir(parents=True, exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            try:
                if isinstance(content, (dict, list)): json.dump(content, f, indent=4, ensure_ascii=False)
                else: f.write(str(content))
            except (TypeError, OverflowError): f.write(str(content))
        print(f"Output successfully saved to {filename}")
    except Exception as e: print(f"Error saving output to file '{filename}': {e}")

# === Testing browser-use with ALL Patches & Custom Controller ===
print("\nTesting browser-use with Patched Agent Methods and Custom Actions:")
try:
    if 'Agent' not in globals() and 'browser_use' in sys.modules: from browser_use import Agent
    google_api_key = os.getenv("GOOGLE_API_KEY");
    if not google_api_key or google_api_key.startswith("your_"): raise ValueError("GOOGLE_API_KEY missing.")
    if 'ChatGoogleGenerativeAI' not in globals(): raise ImportError("ChatGoogleGenerativeAI missing.")

    # Initialize LLMs - Using gemini-2.0-flash for both
    main_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
    planner_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key) # Set consistently
    print(f"Using Main LLM: {getattr(main_llm, 'model', '?')}, Planner LLM configured as: {getattr(planner_llm, 'model', '?')}")
    print("NOTE: _run_planner patch IS ACTIVE, will force main LLM for planning step.") # Clarify patch is active
    print("Note: step patch attempts self-correction and calls planner on error.")

    # --- Define the Task with Vision-Aware Analysis and IFrame Hint ---
    online_editor_vision_task = (
        "1. Navigate to docs.browser-use.com, extract headline and first paragraph. "
        "2. Generate HTML (semantic), CSS (approximate colors based on visual observation of docs.browser-use.com), and JS (button click alters H1). "
        "3. Navigate to https://html-css-js.com/, accept cookies. "
        "4. Attempt to input the generated HTML/CSS/JS into the editor panes (use standard actions or JS recovery using textarea selectors). "
        "5. Attempt to find and click the main 'Run' button. "
        "6. **Use the `extract_iframe_content` action to extract the text content from the output frame.** You will need to provide a CSS selector for the iframe (e.g., `iframe[title='Output']` or perhaps `#result iframe` - inspect the page structure carefully). "
        "7. Analyze the extracted output text: Does it contain the expected headline ('Introduction') and paragraph text? Note discrepancies. "
        "8. Attempt to self-correct the original HTML/CSS/JS based ONLY on clear discrepancies noted in step 7. "
        "9. Use the 'done' action: Report the original HTML/CSS/JS generated, the content extracted (step 6), your analysis (step 7), any corrections attempted (step 8), and difficulties."
    )
    print(f"\nUsing New Vision-Aware Online Editor Task:\n{online_editor_vision_task}\n")
    # ---------------------------------------------


    print("Initializing Agent (with custom controller)...")
    agent = Agent(
        task=online_editor_vision_task, # Use the vision-aware editor task
        llm=main_llm,
        planner_llm=planner_llm, # Pass planner object
        controller=controller, # Pass controller with custom actions
        use_vision=True, # Vision needed for task
        use_vision_for_planner=False, # Patch overrides anyway
        planner_interval=4,
        save_conversation_path="logs/conversation"
    )

    async def run_agent():
        print("Running browser-use agent (with custom actions and planner-on-error)...")
        max_steps_to_run = 150
        print(f"Running agent for maximum {max_steps_to_run} steps...")
        history = await agent.run(max_steps=max_steps_to_run)

        # --- RE-ADDED CHAT LOOP ---
        print("\n--------------------")
        print("Agent run finished.")
        # Check if agent finished successfully based on history
        # Need to access history state carefully
        task_successful = False
        if history and hasattr(history, 'is_successful') and callable(history.is_successful):
            task_successful = history.is_successful()
        elif history and history.history: # Fallback check if last result had error
            last_result_list = history.history[-1].result if history.history else []
            if last_result_list and not any(r.error for r in last_result_list):
                 if history.history[-1].model_output and history.history[-1].model_output.is_done():
                      task_successful = history.history[-1].model_output.get_done_success()


        print(f"Agent task finished {'successfully' if task_successful else 'unsuccessfully'}.")
        print("Starting interactive chat with the LLM...")
        print("Type 'quit' or 'exit' to end chat.")
        print("--------------------")

        chat_history = []
        chat_history.append(SystemMessage(content="You are the LLM that just controlled a browser agent. The user wants to chat with you about the completed task."))
        # Use final_result() method which is safer
        final_agent_report = history.final_result() if history else "Agent did not produce a final result."
        if final_agent_report:
            chat_history.append(AIMessage(content=f"Agent's final report/result was: {final_agent_report}"))

        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ["quit", "exit"]: print("Exiting chat."); break
                if not user_input: continue
                chat_history.append(HumanMessage(content=user_input))
                full_response = ""; print("AI: ", end="", flush=True)
                # Use stream if available, otherwise invoke (checking attribute correctly)
                if hasattr(main_llm, 'astream'):
                    async for chunk in main_llm.astream(chat_history):
                        content_chunk = chunk.content; print(content_chunk, end="", flush=True); full_response += content_chunk
                else:
                    # invoke is sync, use await ainvoke in async func
                    response = await main_llm.ainvoke(chat_history)
                    full_response = response.content; print(full_response, end="")
                print() # Newline after streaming/invoke
                chat_history.append(AIMessage(content=full_response))
            except EOFError: print("\nExiting chat."); break
            except Exception as chat_err: print(f"\nError during chat: {chat_err}")
        # --- END CHAT LOOP ---

        print("\nAgent Run Analysis:")
        steps_taken = agent.state.n_steps -1 if hasattr(agent,'state') else len(history.history)
        print(f"Agent ran for {steps_taken} steps.")
        print("Visited URLs:", history.urls() if history else "N/A")
        # Use final_agent_report captured earlier
        print("Final result type:", type(final_agent_report))
        if final_agent_report is not None: print("Final result:", final_agent_report)
        else: print("Final result: None")
        save_to_file(final_agent_report, filename="agent_output.json")

    if 'asyncio' not in sys.modules: print("ERROR: asyncio module not imported!"); sys.exit(1)
    asyncio.run(run_agent())
    print("\nScript finished.")

except ImportError as e: print(f"ImportError during browser-use test: {e}")
except ValueError as e: print(f"ValueError during browser-use test: {e}")
except Exception as e:
    print(f"Error running browser-use test: {e}")
    print(f"Detailed error for browser-use test:")
    print(f"{traceback.format_exc()}")
# === End of browser-use Test Section ===

# --- Restore Patches ---
if AGENT_CLASS_AVAILABLE:
    target_method_name_planner = "_run_planner"; target_method_name_step = "step"
    if original_agent_run_planner and hasattr(Agent, target_method_name_planner):
        try: setattr(Agent, target_method_name_planner, original_agent_run_planner)
        except Exception as e: print(f"\nFailed to restore Agent patch ({target_method_name_planner}): {e}")
        else: print(f"\nRestored original Agent method: {target_method_name_planner}")
    if original_agent_step and hasattr(Agent, target_method_name_step):
        try: setattr(Agent, target_method_name_step, original_agent_step)
        except Exception as e: print(f"\nFailed to restore Agent patch ({target_method_name_step}): {e}")
        else: print(f"Restored original Agent method: {target_method_name_step}")
# Restore __init__ patch if it was applied
if AGENT_CLASS_AVAILABLE and original_agent_init and hasattr(Agent, '__init__'):
     try: setattr(Agent, '__init__', original_agent_init)
     except Exception as e: print(f"\nFailed to restore Agent patch (__init__): {e}")
     else: print(f"Restored original Agent method: __init__")

if original_playwright_launch:
    try: BrowserType.launch = original_playwright_launch
    except Exception as e: print(f"\nFailed to restore Playwright launch method: {e}")
    else: print("Restored original Playwright launch method.")
# --- End Restore Patches ---