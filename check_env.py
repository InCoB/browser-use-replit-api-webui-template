import os
from dotenv import load_dotenv

load_dotenv()
for k, v in os.environ.items():
    if any(pattern in k for pattern in ['PLAYWRIGHT', 'BROWSER', 'LD_LIBRARY']):
        print(f"{k}={v}")