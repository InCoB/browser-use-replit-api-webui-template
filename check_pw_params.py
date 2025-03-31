#!/usr/bin/env python3
import inspect
from playwright.sync_api import sync_playwright

def main():
    with sync_playwright() as p:
        # Get the signature of the launch method
        launch_sig = inspect.signature(p.chromium.launch)
        print("Playwright chromium.launch() parameters:")
        for param_name, param in launch_sig.parameters.items():
            print(f"  - {param_name}: {param.default}")
            
        # Get the signature of the BrowserType class
        try:
            from playwright._impl._browser_type import BrowserType
            print("\nPlaywright BrowserType class methods:")
            for method_name, method in inspect.getmembers(BrowserType, predicate=inspect.isfunction):
                if method_name.startswith('_'):
                    continue
                print(f"  - {method_name}: {inspect.signature(method)}")
        except ImportError as e:
            print(f"Error importing BrowserType: {e}")

if __name__ == "__main__":
    main()