#!/usr/bin/env python3
"""
Debug monitor for DonutNodes - connects to ComfyUI API for real-time debugging
"""

import requests
import websocket
import json
import uuid
import threading
import time
from typing import Optional, Dict, Any

class ComfyUIDebugMonitor:
    def __init__(self, host="localhost", port=8188):
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws"
        self.client_id = str(uuid.uuid4())
        self.ws: Optional[websocket.WebSocket] = None
        self.monitoring = False
        self.debug_filters = ["SKIP DEBUG", "MERGE DEBUG", "DonutWidenMerge", "LoRADelta"]
        
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        try:
            response = requests.get(f"{self.base_url}/system_stats")
            return response.json()
        except Exception as e:
            print(f"Error getting system stats: {e}")
            return {}
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        try:
            response = requests.get(f"{self.base_url}/queue")
            return response.json()
        except Exception as e:
            print(f"Error getting queue status: {e}")
            return {}
    
    def get_logs(self, raw=False) -> Dict[str, Any]:
        """Get current logs"""
        try:
            endpoint = "/internal/logs/raw" if raw else "/internal/logs"
            response = requests.get(f"{self.base_url}{endpoint}")
            return response.json()
        except Exception as e:
            print(f"Error getting logs: {e}")
            return {}
    
    def search_logs_for_debug(self, patterns=None) -> list:
        """Search logs for debug patterns"""
        if patterns is None:
            patterns = self.debug_filters
            
        logs = self.get_logs(raw=True)
        matching_logs = []
        
        if 'logs' in logs:
            for log_entry in logs['logs']:
                if isinstance(log_entry, str):
                    for pattern in patterns:
                        if pattern in log_entry:
                            matching_logs.append(log_entry)
                            break
        
        return matching_logs
    
    def start_websocket_monitoring(self):
        """Start WebSocket monitoring for real-time updates"""
        try:
            self.ws = websocket.WebSocket()
            self.ws.connect(f"{self.ws_url}?clientId={self.client_id}")
            self.monitoring = True
            
            print(f"[DEBUG MONITOR] WebSocket connected with client ID: {self.client_id}")
            
            while self.monitoring:
                try:
                    message = self.ws.recv()
                    if isinstance(message, str):
                        data = json.loads(message)
                        self._handle_websocket_message(data)
                    else:
                        # Binary message (preview images, etc.)
                        pass
                except websocket.WebSocketTimeoutException:
                    continue
                except Exception as e:
                    print(f"[DEBUG MONITOR] WebSocket error: {e}")
                    break
                    
        except Exception as e:
            print(f"[DEBUG MONITOR] Failed to connect WebSocket: {e}")
        finally:
            if self.ws:
                self.ws.close()
    
    def _handle_websocket_message(self, data: Dict[str, Any]):
        """Handle incoming WebSocket messages"""
        msg_type = data.get('type', '')
        
        if msg_type == 'executing':
            node_info = data.get('data', {})
            if node_info.get('node'):
                node_id = node_info['node']
                print(f"[DEBUG MONITOR] Executing node: {node_id}")
        
        elif msg_type == 'status':
            status = data.get('data', {}).get('status', {})
            if 'exec_info' in status:
                queue_remaining = status['exec_info'].get('queue_remaining', 0)
                if queue_remaining > 0:
                    print(f"[DEBUG MONITOR] Queue remaining: {queue_remaining}")
        
        elif msg_type == 'progress':
            progress_data = data.get('data', {})
            if 'value' in progress_data and 'max' in progress_data:
                value = progress_data['value']
                max_val = progress_data['max']
                print(f"[DEBUG MONITOR] Progress: {value}/{max_val}")
        
        elif msg_type == 'execution_error':
            error_data = data.get('data', {})
            print(f"[DEBUG MONITOR] Execution error: {error_data}")
    
    def stop_monitoring(self):
        """Stop WebSocket monitoring"""
        self.monitoring = False
        if self.ws:
            self.ws.close()
    
    def print_debug_summary(self):
        """Print a summary of current debug information"""
        print("\n" + "="*60)
        print("COMFYUI DEBUG MONITOR SUMMARY")
        print("="*60)
        
        # System stats
        stats = self.get_system_stats()
        if stats:
            system = stats.get('system', {})
            print(f"System: {system.get('os', 'Unknown')} | RAM: {system.get('ram_free', 0):.1f}GB free")
            
            devices = stats.get('devices', [])
            for device in devices:
                if device.get('type') == 'cuda':
                    vram_free = device.get('vram_free', 0) / 1024**3  # Convert to GB
                    vram_total = device.get('vram_total', 0) / 1024**3
                    print(f"GPU {device.get('index', 0)}: {vram_free:.1f}GB free / {vram_total:.1f}GB total")
        
        # Queue status
        queue = self.get_queue_status()
        if queue:
            running = len(queue.get('queue_running', []))
            pending = len(queue.get('queue_pending', []))
            print(f"Queue: {running} running, {pending} pending")
        
        # Recent debug logs
        debug_logs = self.search_logs_for_debug()
        if debug_logs:
            print(f"\nRecent debug messages ({len(debug_logs)} found):")
            for log in debug_logs[-10:]:  # Show last 10
                print(f"  {log.strip()}")
        else:
            print("\nNo recent debug messages found")
        
        print("="*60)

def main():
    """Main function for standalone usage"""
    monitor = ComfyUIDebugMonitor()
    
    print("ComfyUI Debug Monitor")
    print("Commands: 'stats', 'logs', 'debug', 'monitor', 'quit'")
    
    while True:
        try:
            cmd = input("\n> ").strip().lower()
            
            if cmd == 'quit' or cmd == 'q':
                monitor.stop_monitoring()
                break
            elif cmd == 'stats':
                print(json.dumps(monitor.get_system_stats(), indent=2))
            elif cmd == 'logs':
                print(json.dumps(monitor.get_logs(), indent=2))
            elif cmd == 'debug':
                monitor.print_debug_summary()
            elif cmd == 'monitor':
                print("Starting WebSocket monitoring (Ctrl+C to stop)...")
                monitor_thread = threading.Thread(target=monitor.start_websocket_monitoring)
                monitor_thread.daemon = True
                monitor_thread.start()
                try:
                    while monitor.monitoring:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\nStopping monitoring...")
                    monitor.stop_monitoring()
            else:
                print("Unknown command. Available: stats, logs, debug, monitor, quit")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            monitor.stop_monitoring()
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()