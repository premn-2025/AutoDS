#!/usr/bin/env python
"""
API Client Demo - Demonstrates using the AutoDS Pipeline API
"""

import os
import sys
import time
import json
import argparse
import requests
from pathlib import Path


def run_api_demo(
    base_url: str,
    data_file: str,
    target_column: str,
    wait_for_completion: bool = True
):
    """
    Run a pipeline via the API.
    
    Args:
        base_url: API base URL
        data_file: Path to data file
        target_column: Target column name
        wait_for_completion: Whether to poll for completion
    """
    print("🚀 AutoDS Pipeline API Demo")
    print("=" * 50)
    print(f"🌐 API: {base_url}")
    print(f"📁 File: {data_file}")
    print(f"🎯 Target: {target_column}")
    print("=" * 50 + "\n")
    
    # Check health
    print("🔍 Checking API health...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        health = response.json()
        print(f"   Status: {health.get('status', 'unknown')}")
        print(f"   Version: {health.get('version', 'unknown')}")
    except Exception as e:
        print(f"❌ API health check failed: {e}")
        return
    
    # Upload and run pipeline
    print("\n📤 Uploading data and starting pipeline...")
    
    try:
        with open(data_file, 'rb') as f:
            files = {'file': (Path(data_file).name, f)}
            data = {
                'target_column': target_column,
                'config': json.dumps({
                    'test_size': 0.2,
                    'cv_folds': 5
                })
            }
            
            response = requests.post(
                f"{base_url}/pipeline/run",
                files=files,
                data=data,
                timeout=60
            )
        
        if response.status_code != 202:
            print(f"❌ Pipeline start failed: {response.text}")
            return
        
        result = response.json()
        pipeline_id = result.get('pipeline_id')
        print(f"✅ Pipeline started: {pipeline_id}")
        
    except Exception as e:
        print(f"❌ Failed to start pipeline: {e}")
        return
    
    # Wait for completion
    if wait_for_completion:
        print("\n⏳ Waiting for pipeline completion...")
        
        max_wait = 300  # 5 minutes
        interval = 5
        elapsed = 0
        
        while elapsed < max_wait:
            try:
                response = requests.get(
                    f"{base_url}/pipeline/{pipeline_id}/status",
                    timeout=10
                )
                status = response.json()
                
                current_status = status.get('status', 'unknown')
                print(f"   [{elapsed}s] Status: {current_status}")
                
                if current_status == 'completed':
                    print("\n✅ Pipeline completed!")
                    break
                elif current_status == 'failed':
                    print(f"\n❌ Pipeline failed: {status.get('error')}")
                    return
                
                time.sleep(interval)
                elapsed += interval
                
            except Exception as e:
                print(f"   Error checking status: {e}")
                time.sleep(interval)
                elapsed += interval
        
        # Get results
        if elapsed < max_wait:
            print("\n📊 Fetching results...")
            
            try:
                response = requests.get(
                    f"{base_url}/pipeline/{pipeline_id}/result",
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    print("\n" + "=" * 50)
                    print("📈 Pipeline Results")
                    print("=" * 50)
                    print(json.dumps(result.get('result', {}), indent=2, default=str))
                else:
                    print(f"❌ Failed to get results: {response.text}")
                    
            except Exception as e:
                print(f"❌ Error fetching results: {e}")
    
    print("\n🎉 Demo complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='AutoDS Pipeline API Demo')
    
    parser.add_argument(
        '--url',
        type=str,
        default='http://localhost:8080',
        help='API base URL'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to data file'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        required=True,
        help='Target column name'
    )
    
    parser.add_argument(
        '--no-wait',
        action='store_true',
        help='Do not wait for completion'
    )
    
    args = parser.parse_args()
    
    run_api_demo(
        base_url=args.url,
        data_file=args.data,
        target_column=args.target,
        wait_for_completion=not args.no_wait
    )


if __name__ == '__main__':
    main()
