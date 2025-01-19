"""
A script to set up the virtual environment with required dependencies for app.py.

Author: Angel Rivera
Date: 01/19/2025
"""

import os
import subprocess
import sys

def setup():
    # Step 1: Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(current_dir, "venv")
    python_executable = os.path.join(venv_dir, "Scripts", "python") if os.name == "nt" else os.path.join(venv_dir, "bin", "python")

    # Step 2: Create the virtual environment if it doesn't exist
    if not os.path.exists(venv_dir):
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
        print("Virtual environment created successfully!")

    # Step 4: Activate the virtual environment and install dependencies
    print("Installing dependencies...")
    try:
        subprocess.check_call([python_executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([python_executable, "-m", "pip", "install", 
            "tensorflow", "opencv-python", "mediapipe", "scikit-learn", "matplotlib"])
        print("Dependencies installed successfully!")
    except Exception as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)
    
    # Step 5: Notify the user how to activate the virtual environment
    print("\nSetup completed successfully!")
    print("To activate the virtual environment, run:")
    if os.name == "nt":  # Windows
        print(f"    {os.path.join(venv_dir, 'Scripts', 'activate')}")
    else:  # macOS/Linux
        print(f"    source {os.path.join(venv_dir, 'bin', 'activate')}")

if __name__ == "__main__":
    setup()
