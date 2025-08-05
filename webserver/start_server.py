#!/usr/bin/env python3
"""
VITON-HD Web Server Startup Script
"""

import os
import sys
import uvicorn
from pathlib import Path

def main():
    # Change to webserver directory
    webserver_dir = Path(__file__).parent
    os.chdir(webserver_dir)
    
    print("🚀 Starting VITON-HD Web Server...")
    print(f"📁 Working directory: {webserver_dir}")
    print("🌐 Server will be available at: http://localhost:8000")
    print("📝 Upload person and cloth images to generate virtual try-on results")
    print("-" * 60)
    
    # Check if required directories exist
    required_dirs = [
        "static/uploads",
        "static/results", 
        "templates",
        "../checkpoints"
    ]
    
    for dir_path in required_dirs:
        full_path = webserver_dir / dir_path
        if not full_path.exists():
            print(f"❌ Missing directory: {full_path}")
            if "checkpoints" in dir_path:
                print("   Make sure your checkpoints are in the ../checkpoints/ directory")
                print("   Required files: seg_final.pth, gmm_final.pth, alias_final.pth")
            else:
                print(f"   Creating directory: {full_path}")
                full_path.mkdir(parents=True, exist_ok=True)
    
    # Check for checkpoint files
    checkpoint_files = [
        "../checkpoints/seg_final.pth",
        "../checkpoints/gmm_final.pth", 
        "../checkpoints/alias_final.pth"
    ]
    
    missing_checkpoints = []
    for ckpt in checkpoint_files:
        if not (webserver_dir / ckpt).exists():
            missing_checkpoints.append(ckpt)
    
    if missing_checkpoints:
        print("⚠️  Warning: Missing checkpoint files:")
        for ckpt in missing_checkpoints:
            print(f"   - {ckpt}")
        print("   The server will start but may not work correctly without these files.")
        print()
    
    try:
        # Start the server
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()