import os
import requests
import time
from dotenv import load_dotenv
import sys
import json

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Remove detection/construction logic
# BASE_URL = os.environ.get("REPL_EXTERNAL_URL") 
# ... (Removed fallback logic) ...

#

# Read Base URL and API Key from environment variables
BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("EXTERNAL_API_KEY")

# Simple task for testing
TEST_TASK_DESCRIPTION = "Go to example.com and return the main heading text." 
# Optional: Specify a model, otherwise API default (currently gpt-4o) will be used
# TEST_MODEL = "gpt-4o" 
TEST_MODEL = None # Use API default

# --- Check Configuration ---
# Add check for API_BASE_URL
if not BASE_URL or BASE_URL == "your_repl_external_url_here":
    print("Error: API_BASE_URL not found or not set in .env file.")
    print("Please set the base URL of your running Repl in the .env file.")
    sys.exit(1)
    
if not API_KEY or API_KEY == "your_internal_api_secret_key_here":
    print("Error: EXTERNAL_API_KEY not found or not set in .env file.")
    print("Please set your secret API key in the .env file.")
    sys.exit(1)

# Construct URLs and Headers
API_URL = f"{BASE_URL.rstrip('/')}/api/browser-tasks"
DIAGNOSTICS_URL = f"{BASE_URL.rstrip('/')}/diagnostics" # New URL for diagnostics
HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}

# --- New Diagnostics Test Function ---
def test_diagnostics_endpoint():
    print("--- Testing /diagnostics Endpoint --- ")
    try:
        print(f"Calling GET {DIAGNOSTICS_URL}...")
        response = requests.get(DIAGNOSTICS_URL, headers=HEADERS, timeout=60) # Add timeout
        
        # Check if the call itself was successful (200 OK or 503 Service Unavailable are expected)
        if response.status_code == 200:
            print(f"Diagnostics call successful (Status Code: {response.status_code})")
        elif response.status_code == 503:
             print(f"Diagnostics call successful, API reported unhealthy (Status Code: {response.status_code})")
        else:
            # Raise an exception for other unexpected status codes
            response.raise_for_status() 
            
        # Print the detailed diagnostics report
        print("Diagnostics Report:")
        try:
            diagnostics_data = response.json()
            print(json.dumps(diagnostics_data, indent=2))
            # Optionally, log the reported overall status
            overall_status = diagnostics_data.get("overall_status", "unknown")
            print(f"Reported Overall Status: {overall_status}")
        except json.JSONDecodeError:
            print("Error: Could not decode JSON response from /diagnostics")
            print(f"Raw response text: {response.text[:500]}...") # Print first 500 chars
            
    except requests.exceptions.Timeout:
        print("Error: Request to /diagnostics timed out.")
    except requests.exceptions.RequestException as e:
        print(f"Error calling /diagnostics endpoint: {e}")
        if e.response is not None:
            try:
                print(f"Response Status Code: {e.response.status_code}")
                print(f"Response Body: {e.response.text[:500]}...")
            except:
                pass
    except Exception as e:
        print(f"An unexpected error occurred during diagnostics test: {e}")
    finally:
        print("--- Diagnostics Test Finished --- \n")
# --- End Diagnostics Test Function ---

print(f"--- API Test Script ---")
# Update print statement to use BASE_URL variable
print(f"API Endpoint Target Base: {BASE_URL}") 
print(f"Using API Key: ...{API_KEY[-4:]}") # Show only last 4 chars for confirmation
print(f"Task: {TEST_TASK_DESCRIPTION}")
if TEST_MODEL:
    print(f"Model: {TEST_MODEL}")
else:
    print("Model: Using API default")
print("-----------------------")

# --- Run Diagnostics Test First ---
test_diagnostics_endpoint() 

# --- 1. Create Task ---
task_id = None
initial_status = None
try:
    print("Attempting to create task...")
    payload = {"task": TEST_TASK_DESCRIPTION}
    if TEST_MODEL:
        payload["model"] = TEST_MODEL
        
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
    
    data = response.json()
    task_id = data.get("id")
    initial_status = data.get("status")
    
    # Accept 'pending' or 'completed' as valid initial status for fast tasks
    if not task_id or initial_status not in ["pending", "completed"]:
        print(f"Error: Unexpected initial status in response: {data}")
        sys.exit(1)

    print(f"Task created successfully! ID: {task_id}, Initial Status: {initial_status}")

except requests.exceptions.RequestException as e:
    print(f"Error creating task: {e}")
    if e.response is not None:
        try:
            print(f"Response Status Code: {e.response.status_code}")
            print(f"Response Body: {e.response.text}")
        except:
            pass
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during task creation: {e}")
    sys.exit(1)
    
# --- 2. Poll Until Task Finishes ---
final_status = initial_status
polling_error = None

if initial_status == "pending":
    print(f"Polling status for task {task_id} until finished...")
    polling_attempts = 0
    max_polling_attempts = 60 # Poll for max 2 minutes
    while polling_attempts < max_polling_attempts:
        polling_attempts += 1
        try:
            time.sleep(2)
            poll_url = f"{API_URL}/{task_id}"
            response = requests.get(poll_url, headers=HEADERS)
            response.raise_for_status()
            data = response.json()
            current_status = data.get("status")
            print(f"  Attempt {polling_attempts}: Status = {current_status}")
            if current_status in ["completed", "failed"]:
                final_status = current_status
                break # Exit loop
        except requests.exceptions.RequestException as e:
            print(f"  Attempt {polling_attempts}: Error polling status: {e}")
            if e.response is not None and e.response.status_code == 404:
                 print(f"  Task ID {task_id} not found. Stopping poll.")
                 final_status = "error"
                 polling_error = "Task ID not found during polling (404)."
                 break
            # Continue polling on other request errors for now
        except Exception as e:
            print(f"  Attempt {polling_attempts}: An unexpected error during polling: {e}")
            final_status = "error"
            polling_error = f"Unexpected polling error: {e}"
            break # Stop polling on unexpected errors
    else: # Loop finished without break
        print(f"Max polling attempts ({max_polling_attempts}) reached.")
        final_status = "timeout"

# --- 3. Fetch Final Result/Error Explicitly ---
print("-----------------------")
final_result_data = None

if final_status == "completed":
    print(f"Task {task_id} completed. Fetching final result...")
    try:
        poll_url = f"{API_URL}/{task_id}"
        response = requests.get(poll_url, headers=HEADERS)
        response.raise_for_status()
        final_result_data = response.json()
        print("Successfully fetched final task data.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching final result: {e}")
        final_status = "error_fetching_result"
        polling_error = polling_error or f"Could not fetch final result: {e}"
    except Exception as e:
        print(f"Unexpected error fetching final result: {e}")
        final_status = "error_fetching_result"
        polling_error = polling_error or f"Unexpected error fetching final result: {e}"
elif final_status == "failed":
     print(f"Task {task_id} failed. Attempting to fetch final error details...")
     try:
        poll_url = f"{API_URL}/{task_id}"
        response = requests.get(poll_url, headers=HEADERS)
        response.raise_for_status()
        final_result_data = response.json() # Get the final state including the error field
        polling_error = final_result_data.get("error") # Extract error from final data
     except Exception as e:
         print(f"Warning: Could not fetch final details for failed task: {e}")
         polling_error = polling_error or "(Could not fetch error details for failed task)"

# --- 4. Print Final Outcome ---
print(f"Final Task Status: {final_status}")

# Print the raw data fetched in step 3 if available
if final_result_data:
    print("Raw Fetched Data:")
    try:
        print(json.dumps(final_result_data, indent=2))
    except Exception as json_err:
        print(f"(Could not pretty-print JSON: {json_err}) Raw: {final_result_data}")
    print("-----")

if final_status == "completed" and final_result_data:
    print("Extracted Result:")
    extracted_result = final_result_data.get("result")
    if extracted_result is not None:
        print(extracted_result)
    else:
        print("(Result field not found or is null in the fetched data)")
elif final_status == "failed":
    print("Error Details:")
    print(polling_error) # Print error fetched in step 3 or during polling
elif final_status == "timeout":
    print("Task timed out before completion.")
elif final_status == "error_fetching_result":
     print("Task completed, but failed to fetch the final result object.")
     print(f"Details: {polling_error}")
else: # Includes initial 'error' status from polling etc.
    print("Task did not complete successfully.")
    if polling_error:
        print(f"Details: {polling_error}")

print("--- Test Complete ---")

# Exit with appropriate code
if final_status == "completed":
    sys.exit(0)
else:
    sys.exit(1) 