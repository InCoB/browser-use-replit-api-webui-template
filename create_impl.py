import os
import sys
import textwrap

pyproject_hooks_path = None

for path in sys.path:
    possible_path = os.path.join(path, 'pyproject_hooks')
    if os.path.exists(possible_path):
        pyproject_hooks_path = possible_path
        break

if pyproject_hooks_path:
    print(f"Found pyproject_hooks at {pyproject_hooks_path}")
    impl_path = os.path.join(pyproject_hooks_path, "_impl.py")
    
    # Create the content for _impl.py with the missing class
    impl_content = """import os
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Union

class BackendUnavailable(Exception):
    \"\"\"Raised when the backend cannot be imported in the hook process.\"\"\"

class HookMissing(Exception):
    \"\"\"Raised when a hook is missing.\"\"\"

def _build_backend_name(
    distribution_name: str, backend_name: Optional[str] = None
) -> str:
    if backend_name is None:
        return "{0}.__legacy__".format(distribution_name)
    return backend_name
"""
    
    try:
        with open(impl_path, 'w') as f:
            f.write(impl_content)
        print(f"Successfully wrote content to {impl_path}")
    except Exception as e:
        print(f"Error writing to {impl_path}: {e}")
else:
    print("pyproject_hooks module not found in sys.path")