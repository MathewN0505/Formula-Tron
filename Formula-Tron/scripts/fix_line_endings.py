#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Ensure required packages are installed."""
    required = ['PyQt5', 'scipy', 'cv2', 'numpy']
    missing = []
    
    for pkg in required:
        try:
            if pkg == 'cv2':
                import cv2
            else:
                __import__(pkg)
        except ImportError:
            missing.append(pkg)
            
    if missing:
        print(f"Installing missing dependencies: {', '.join(missing)}...")
        # Map python names to apt package names if possible, otherwise rely on pre-installed env
        # Since we can't easily sudo install from python script without password,
        # we will try pip as user or warn
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        except:
            print(f"Warning: Could not install {missing}. If code crashes, install them manually.")

def fix_directory(directory):
    extensions = {'.py', '.sh', '.yaml', '.yml', '.xml', '.txt', '.md', '.launch.py'}
    fixed = 0
    
    for filepath in Path(directory).rglob('*'):
        if filepath.is_dir() or filepath.name.startswith('.'):
            continue
        if any(part in filepath.parts for part in ['build', 'install', 'log', '__pycache__']):
            continue
        
        if filepath.suffix in extensions or str(filepath).endswith('.launch.py'):
            try:
                with open(filepath, 'rb') as f:
                    content = f.read()
                if b'\r\n' in content:
                    with open(filepath, 'wb') as f:
                        f.write(content.replace(b'\r\n', b'\n'))
                    fixed += 1
            except:
                pass
    
    return fixed

if __name__ == '__main__':
    # 1. Check dependencies first
    check_dependencies()
    
    # 2. Fix line endings
    script_dir = Path(__file__).parent
    fixed = fix_directory(script_dir)
    sys.exit(0)
