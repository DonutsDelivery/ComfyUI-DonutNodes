#!/usr/bin/env python3
"""Simple WebSocket monitor for ComfyUI API logs and events"""

import asyncio
import websockets
import json
import sys
from datetime import datetime

async def monitor_websocket():
    """Connect to ComfyUI WebSocket and monitor for WIDEN-related messages"""
    uri = "ws://localhost:8188/ws?clientId=widen_monitor"
    
    try:
        print(f"[{datetime.now()}] Connecting to ComfyUI WebSocket...")
        
        async with websockets.connect(uri) as websocket:
            print(f"[{datetime.now()}] Connected! Monitoring for WIDEN merge activity...")
            print("=" * 60)
            
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    # Print all messages for debugging
                    print(f"[{datetime.now()}] Received: {data.get('type', 'unknown')}")
                    
                    # Look for execution progress or logs that might contain WIDEN info
                    if 'data' in data:
                        msg_data = data['data']
                        
                        # Check for node execution updates
                        if 'node' in msg_data:
                            node_info = msg_data.get('node', '')
                            if 'widen' in str(node_info).lower() or 'merge' in str(node_info).lower():
                                print(f"[WIDEN ACTIVITY] Node: {node_info}")
                        
                        # Check for any text output that might contain our debug messages
                        if 'output' in msg_data:
                            output = str(msg_data['output'])
                            if any(keyword in output.lower() for keyword in ['widen', 'compatibility', 'score debug', 'enhanced widen']):
                                print(f"[WIDEN OUTPUT] {output}")
                    
                    # Print full message if it might be WIDEN-related
                    full_msg = str(data).lower()
                    if any(keyword in full_msg for keyword in ['widen', 'merge', 'compatibility', 'donut']):
                        print(f"[FULL MESSAGE] {json.dumps(data, indent=2)}")
                        
                except websockets.exceptions.ConnectionClosed:
                    print(f"[{datetime.now()}] WebSocket connection closed")
                    break
                except json.JSONDecodeError:
                    print(f"[{datetime.now()}] Received non-JSON message: {message}")
                except Exception as e:
                    print(f"[{datetime.now()}] Error: {e}")
                    
    except Exception as e:
        print(f"[{datetime.now()}] Failed to connect: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("WIDEN WebSocket Monitor - Press Ctrl+C to exit")
    try:
        asyncio.run(monitor_websocket())
    except KeyboardInterrupt:
        print("\n[Monitor] Exiting...")