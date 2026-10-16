#!/usr/bin/env python
"""
Simple script to start the AutoDS Pipeline server
"""
import sys
import os

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 AutoDS Pipeline - Starting Production Server")
    print("=" * 60)
    
    try:
        from app import app
        
        port = int(os.getenv('PORT', 8080))
        host = os.getenv('HOST', '0.0.0.0')
        
        print(f"\n📡 Server starting on http://{host}:{port}")
        print(f"📋 Health check: http://localhost:{port}/health")
        print(f"📋 API Docs: http://localhost:{port}/config")
        print("\n🔧 Available endpoints:")
        print("   POST /pipeline/run     - Run ML pipeline on uploaded data")
        print("   GET  /pipeline/{id}/status - Check pipeline status")
        print("   GET  /pipeline/{id}/result - Get pipeline results")
        print("   GET  /health           - Health check")
        print("   GET  /metrics          - View metrics")
        print("\n" + "=" * 60)
        print("Press Ctrl+C to stop the server")
        print("=" * 60 + "\n")
        
        app.run(host=host, port=port, debug=False, threaded=True)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)
