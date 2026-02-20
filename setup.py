#!/usr/bin/env python3
"""
Setup script for Luccia - Emotion-Driven Art Generator
=====================================================

This script helps users set up the Luccia environment and install dependencies.

Author: Luccia Development Team
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Print the Luccia banner."""
    print("=" * 60)
    print("Luccia - Emotion-Driven Art Generator")
    print("Setup and Installation Script")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True

def install_package(package_name, pip_name=None):
    """Install a package using pip."""
    if pip_name is None:
        pip_name = package_name
    
    try:
        print(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
        print(f"✅ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}: {e}")
        return False

def check_package(package_name):
    """Check if a package is installed."""
    try:
        __import__(package_name)
        print(f"✅ {package_name} - OK")
        return True
    except ImportError:
        print(f"❌ {package_name} - Not installed")
        return False

def install_requirements():
    """Install required packages."""
    print("\nInstalling required packages...")
    
    required_packages = [
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("pygame", "pygame"),
        ("pillow", "PIL")
    ]
    
    optional_packages = [
        ("dlib", "dlib"),
        ("tensorflow", "tensorflow"),
        ("torch", "torch")
    ]
    
    # Install required packages
    for pip_name, import_name in required_packages:
        if not check_package(import_name):
            if not install_package(import_name, pip_name):
                print(f"⚠️  Warning: {import_name} installation failed")
    
    # Check optional packages
    print("\nChecking optional packages...")
    for pip_name, import_name in optional_packages:
        check_package(import_name)

def download_dlib_model():
    """Download dlib facial landmark model."""
    model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    model_file = "shape_predictor_68_face_landmarks.dat"
    
    if os.path.exists(model_file):
        print(f"✅ Facial landmark model already exists: {model_file}")
        return True
    
    print(f"\nDownloading dlib facial landmark model...")
    print(f"URL: {model_url}")
    print("This model is required for advanced facial landmark detection.")
    print("You can download it manually and place it in the project directory.")
    
    try:
        import urllib.request
        print("Attempting automatic download...")
        urllib.request.urlretrieve(model_url, model_file + ".bz2")
        
        # Decompress the file
        import bz2
        with bz2.open(model_file + ".bz2", 'rb') as source, open(model_file, 'wb') as target:
            target.write(source.read())
        
        # Remove compressed file
        os.remove(model_file + ".bz2")
        
        print(f"✅ Facial landmark model downloaded: {model_file}")
        return True
    except Exception as e:
        print(f"❌ Automatic download failed: {e}")
        print("Please download the model manually from:")
        print(f"  {model_url}")
        print("Extract it and place 'shape_predictor_68_face_landmarks.dat' in the project directory.")
        return False

def test_installation():
    """Test the installation by running a simple test."""
    print("\nTesting installation...")
    
    try:
        # Test basic imports
        import cv2
        import numpy as np
        import pygame
        print("✅ Basic imports successful")
        
        # Test camera access
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera access successful")
            cap.release()
        else:
            print("⚠️  Camera not available (will use simulated feed)")
        
        # Test pygame initialization
        pygame.init()
        pygame.quit()
        print("✅ Pygame initialization successful")
        
        return True
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    directories = ["saved_art", "logs"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def main():
    """Main setup function."""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    install_requirements()
    
    # Download dlib model
    download_dlib_model()
    
    # Create directories
    create_directories()
    
    # Test installation
    if test_installation():
        print("\n🎉 Setup completed successfully!")
        print("\nTo run Luccia:")
        print("  python luccia.py")
        print("\nFor help:")
        print("  python luccia.py --help")
    else:
        print("\n⚠️  Setup completed with warnings.")
        print("Some features may not work properly.")
        print("Check the error messages above for details.")

if __name__ == "__main__":
    main()
