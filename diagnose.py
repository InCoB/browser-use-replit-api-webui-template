import sys
import os

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print("Python path:")
for p in sys.path:
    print(f"  {p}")

try:
    import pyproject_hooks
    print(f"pyproject_hooks version: {pyproject_hooks.__version__}")
    print(f"pyproject_hooks path: {pyproject_hooks.__file__}")
    print("Contents of pyproject_hooks._impl:")
    try:
        with open(os.path.join(os.path.dirname(pyproject_hooks.__file__), "_impl.py"), "r") as f:
            print(f.read())
    except Exception as e:
        print(f"Error reading _impl.py: {e}")
except ImportError as e:
    print(f"Import error: {e}")