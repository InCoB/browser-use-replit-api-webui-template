import os
import sys

pyproject_hooks_path = None

for path in sys.path:
    possible_path = os.path.join(path, 'pyproject_hooks')
    if os.path.exists(possible_path):
        pyproject_hooks_path = possible_path
        break

if pyproject_hooks_path:
    print(f"Found pyproject_hooks at {pyproject_hooks_path}")
    impl_path = os.path.join(pyproject_hooks_path, "_impl.py")
    if os.path.exists(impl_path):
        print(f"Found _impl.py at {impl_path}")
        with open(impl_path, 'r') as f:
            content = f.read()
        print("\nContents of _impl.py:")
        print(content)
    else:
        print(f"_impl.py not found at {impl_path}")
else:
    print("pyproject_hooks module not found in sys.path")