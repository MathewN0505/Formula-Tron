import sys
import traceback
import os

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pov_driving_sim
except Exception as e:
    with open("dll_error.log", "w") as f:
        traceback.print_exc(file=f)
    print("Error saved to dll_error.log")
