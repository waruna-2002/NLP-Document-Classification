import sys
import os

# Resolve absolute paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, 'src')

# Add the 'src' directory to Python path so that internal modules can be imported
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Path to the actual app.py
src_app_path = os.path.join(SRC_DIR, 'app.py')

# Read and execute src/app.py, overriding __file__ to preserve relative paths
with open(src_app_path, 'r', encoding='utf-8') as f:
    code = f.read()

globals_dict = {
    "__file__": src_app_path,
    "__name__": "__main__",
}
exec(code, globals_dict)
