"""
Donut Prompt Receiver Node
Receives prompts via HTTP and queues the current workflow.
"""

import json
import threading
import random
import os
from http.server import HTTPServer, BaseHTTPRequestHandler

import urllib.request
import urllib.error


class PromptReceiverServer:
    """HTTP server that receives prompts and queues the workflow."""

    _instance = None
    _server = None
    _thread = None
    _prompt_queue = []  # FIFO queue for prompts
    _api_queued = 0     # Count of API-triggered workflows queued
    _api_processed = 0  # Count of API-triggered workflows processed
    _workflow = None
    _node_id = None
    _results = {}       # prompt_id -> {"image_path": path, "ready": bool}
    _reporter_node_id = None  # Node ID of DonutImageReporter
    _current_prompt_id = None  # Current prompt_id being processed (fallback for reporter)

    # Loop feature
    _loop_enabled = False
    _last_action = None  # "send_to_claude" or "received_from_api"
    _last_send_config = None  # Dict with user_prompt, meta_prompt, num_prompts, outgoing config
    _queue_monitor_thread = None
    _queue_was_busy = False  # Track if queue had items

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, workflow, node_id):
        """Store the current workflow and node ID."""
        self._workflow = workflow
        self._node_id = node_id

    def register_reporter(self, node_id):
        """Register the reporter node ID."""
        self._reporter_node_id = node_id

    def store_result(self, prompt_id, image_path):
        """Store a completed result."""
        existing = self._results.get(prompt_id, {})
        existing["image_path"] = image_path
        existing["ready"] = True
        self._results[prompt_id] = existing
        print(f"[DonutPromptReceiver] Result stored for {prompt_id}: {image_path}")

    def store_prompt_text(self, prompt_id, prompt_text):
        """Store the prompt text for a queued workflow."""
        self._results[prompt_id] = {"prompt": prompt_text, "ready": False}

    def get_prompt_text(self, prompt_id):
        """Get the prompt text for a prompt_id."""
        result = self._results.get(prompt_id, {})
        return result.get("prompt", "")

    def get_result(self, prompt_id):
        """Get a stored result."""
        return self._results.get(prompt_id)

    def set_loop_enabled(self, enabled):
        """Enable or disable loop mode."""
        self._loop_enabled = enabled
        # Clear busy flag when disabling to prevent accidental triggers
        if not enabled:
            self._queue_was_busy = False
        print(f"[DonutPromptReceiver] Loop {'enabled' if enabled else 'disabled'}")
        if enabled and self._queue_monitor_thread is None:
            self._start_queue_monitor()

    def set_last_send_config(self, config):
        """Store the last 'Send to Claude' configuration."""
        self._last_send_config = config
        self._last_action = "send_to_claude"
        # Only mark busy if loop is enabled
        if self._loop_enabled:
            self._queue_was_busy = True
        print(f"[DonutPromptReceiver] Stored send_to_claude config")

    def mark_api_received(self, is_external=False):
        """Mark that we received prompts from API.

        Args:
            is_external: If True, this is from an external source (Discord, etc.)
                        and should override any previous "send_to_claude" action.
        """
        if is_external:
            # External source (Discord, etc.) - always update
            self._last_action = "received_from_api"
        elif self._last_action != "send_to_claude":
            # Only update if not expecting Claude's response
            self._last_action = "received_from_api"
        # Only mark busy if loop is enabled
        if self._loop_enabled:
            self._queue_was_busy = True

    def _start_queue_monitor(self):
        """Start the queue monitor thread."""
        if self._queue_monitor_thread is not None:
            return

        import time

        def monitor():
            while True:
                time.sleep(3)  # Check every 3 seconds
                if not self._loop_enabled:
                    continue

                try:
                    # Check ComfyUI queue status
                    req = urllib.request.Request("http://127.0.0.1:8188/queue")
                    with urllib.request.urlopen(req, timeout=2) as response:
                        queue_data = json.loads(response.read().decode('utf-8'))
                        running = queue_data.get("queue_running", [])
                        pending = queue_data.get("queue_pending", [])
                        queue_empty = len(running) == 0 and len(pending) == 0

                        if queue_empty and self._queue_was_busy and self._loop_enabled:
                            # Queue just became empty after having work - double check loop still enabled
                            print(f"[DonutPromptReceiver] Queue empty, loop enabled, triggering re-send...")
                            self._trigger_loop()
                            self._queue_was_busy = False
                        elif not queue_empty and self._loop_enabled:
                            self._queue_was_busy = True

                except Exception as e:
                    # Queue check failed, ignore
                    pass

        self._queue_monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._queue_monitor_thread.start()
        print("[DonutPromptReceiver] Queue monitor started")

    def _trigger_loop(self):
        """Trigger the loop action based on last action type."""
        # Final safety check - only trigger if loop is actually enabled
        if not self._loop_enabled:
            print("[DonutPromptReceiver] Loop disabled, not triggering")
            return

        if not self._last_send_config:
            print("[DonutPromptReceiver] No config stored, cannot loop")
            return

        config = self._last_send_config
        outgoing_url = f"{config['protocol']}://{config['host']}:{config['port']}/prompt"

        if self._last_action == "send_to_claude":
            # Re-send the same user_prompt + meta_prompt to Claude
            port = config.get('incoming_port', 8001)
            api_info = f"""
ComfyUI Prompt API (port {port}):
- POST http://localhost:{port}/prompt with JSON body
- Single: {{"prompt": "main prompt", "face_prompt": "face details", "negative_prompt": "bad things"}}
- Multiple: {{"prompts": [{{"prompt": "...", "face_prompt": "...", "negative_prompt": "..."}}, ...]}}
- face_prompt and negative_prompt are optional
"""
            full_prompt = f"""{api_info}

Apply this meta prompt: "{config['meta_prompt']}"
To this base prompt: "{config['user_prompt']}"

Generate {config['num_prompts']} unique variation(s) and queue them using the API above."""

            print(f"[DonutPromptReceiver] Loop: Re-sending to Claude (send_to_claude)")
        else:
            # Last action was receiving from API - ask Claude to redo
            full_prompt = "Redo the last prompt I gave you - generate the same number of variations and queue them."
            print(f"[DonutPromptReceiver] Loop: Asking Claude to redo (received_from_api)")

        try:
            data = json.dumps({"prompt": full_prompt}).encode('utf-8')
            req = urllib.request.Request(
                outgoing_url,
                data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                print(f"[DonutPromptReceiver] Loop: Sent to {outgoing_url}")
        except Exception as e:
            print(f"[DonutPromptReceiver] Loop: Failed to send to {outgoing_url}: {e}")

    def start(self, port=8001):
        if self._server is not None:
            return

        receiver = self

        class RequestHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                print(f"[DonutPromptReceiver] {args[0]}")

            def _send_json(self, status, data):
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())

            def do_OPTIONS(self):
                self.send_response(200)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                self.end_headers()

            def do_POST(self):
                if self.path == "/register":
                    content_length = int(self.headers.get('Content-Length', 0))
                    body = self.rfile.read(content_length).decode('utf-8')
                    try:
                        data = json.loads(body)
                        workflow = data.get("workflow")
                        node_id = data.get("node_id")
                        if workflow and node_id:
                            receiver.register(workflow, node_id)
                            self._send_json(200, {"status": "workflow registered"})
                        else:
                            self._send_json(400, {"error": "Missing workflow or node_id"})
                    except Exception as e:
                        self._send_json(500, {"error": str(e)})
                elif self.path == "/reset":
                    receiver._workflow = None
                    receiver._prompt_queue = []
                    receiver._api_queued = 0
                    receiver._api_processed = 0
                    self._send_json(200, {"status": "reset"})
                elif self.path == "/cancel" or self.path == "/cancel_all":
                    try:
                        # Interrupt current
                        req = urllib.request.Request("http://127.0.0.1:8188/interrupt", method="POST")
                        urllib.request.urlopen(req)
                        # Clear queue
                        req = urllib.request.Request(
                            "http://127.0.0.1:8188/queue",
                            data=json.dumps({"clear": True}).encode('utf-8'),
                            headers={"Content-Type": "application/json"}
                        )
                        urllib.request.urlopen(req)
                        self._send_json(200, {"status": "cancelled all"})
                    except Exception as e:
                        self._send_json(500, {"error": str(e)})
                elif self.path == "/interrupt" or self.path == "/skip":
                    try:
                        req = urllib.request.Request("http://127.0.0.1:8188/interrupt", method="POST")
                        urllib.request.urlopen(req)
                        self._send_json(200, {"status": "interrupted"})
                    except Exception as e:
                        self._send_json(500, {"error": str(e)})
                elif self.path == "/clear":
                    try:
                        req = urllib.request.Request(
                            "http://127.0.0.1:8188/queue",
                            data=json.dumps({"clear": True}).encode('utf-8'),
                            headers={"Content-Type": "application/json"}
                        )
                        urllib.request.urlopen(req)
                        self._send_json(200, {"status": "queue cleared"})
                    except Exception as e:
                        self._send_json(500, {"error": str(e)})
                elif self.path == "/loop/config":
                    content_length = int(self.headers.get('Content-Length', 0))
                    body = self.rfile.read(content_length).decode('utf-8')
                    try:
                        data = json.loads(body)
                        enabled = data.get("enabled")
                        if enabled is not None:
                            # Handle string "true"/"false" from JS as well as actual booleans
                            if isinstance(enabled, str):
                                enabled = enabled.lower() == "true"
                            receiver.set_loop_enabled(bool(enabled))

                        # Store send_to_claude config if provided
                        if data.get("user_prompt") is not None:
                            config = {
                                "user_prompt": data.get("user_prompt", ""),
                                "meta_prompt": data.get("meta_prompt", ""),
                                "num_prompts": data.get("num_prompts", 1),
                                "host": data.get("host", "localhost"),
                                "port": data.get("port", 5728),
                                "protocol": data.get("protocol", "http"),
                                "incoming_port": data.get("incoming_port", 8001),
                            }
                            receiver.set_last_send_config(config)

                        self._send_json(200, {
                            "status": "ok",
                            "loop_enabled": receiver._loop_enabled,
                            "last_action": receiver._last_action
                        })
                    except Exception as e:
                        self._send_json(500, {"error": str(e)})
                elif self.path == "/prompt" or self.path.startswith("/prompt?"):
                    content_length = int(self.headers.get('Content-Length', 0))
                    body = self.rfile.read(content_length).decode('utf-8')

                    try:
                        data = json.loads(body)

                        # Mark that we received prompts from API
                        # If "external" flag is set, this overrides any "send_to_claude" state
                        is_external = data.get("external", False)
                        receiver.mark_api_received(is_external=is_external)

                        # Support single prompt or array of prompts
                        prompts = data.get("prompts", [])
                        single_prompt = data.get("prompt", "")

                        # Get optional prompts for single prompt mode
                        face_prompt = data.get("face_prompt", "")
                        negative_prompt = data.get("negative_prompt", "")

                        if single_prompt and not prompts:
                            prompts = [{"prompt": single_prompt, "face_prompt": face_prompt, "negative_prompt": negative_prompt}]
                        elif prompts:
                            # Normalize prompts array - strings become dicts
                            normalized = []
                            for p in prompts:
                                if isinstance(p, str):
                                    normalized.append({"prompt": p})
                                else:
                                    normalized.append(p)
                            prompts = normalized

                        if not prompts:
                            self._send_json(400, {"error": "No prompt(s) provided"})
                            return

                        results = []
                        for p in prompts:
                            prompt_text = p.get("prompt", "") if isinstance(p, dict) else p
                            fp = p.get("face_prompt", "") if isinstance(p, dict) else ""
                            np = p.get("negative_prompt", "") if isinstance(p, dict) else ""
                            receiver._prompt_queue.append(prompt_text)
                            receiver._api_queued += 1
                            result = receiver.queue_workflow(prompt_text, fp, np)
                            results.append(result)

                        # Return single result for single prompt, array for multiple
                        if len(results) == 1:
                            self._send_json(200, results[0])
                        else:
                            self._send_json(200, {"queued": len(results), "results": results})
                    except json.JSONDecodeError:
                        self._send_json(400, {"error": "Invalid JSON"})
                    except Exception as e:
                        self._send_json(500, {"error": str(e)})
                else:
                    self._send_json(404, {"error": "Not found"})

            def do_GET(self):
                if self.path == "/health":
                    self._send_json(200, {
                        "status": "running",
                        "workflow_registered": receiver._workflow is not None,
                        "reporter_registered": receiver._reporter_node_id is not None,
                        "pending_prompts": len(receiver._prompt_queue)
                    })
                elif self.path == "/loop/status":
                    self._send_json(200, {
                        "loop_enabled": receiver._loop_enabled,
                        "last_action": receiver._last_action,
                        "queue_was_busy": receiver._queue_was_busy,
                        "has_config": receiver._last_send_config is not None,
                        "monitor_running": receiver._queue_monitor_thread is not None
                    })
                elif self.path == "/queue" or self.path == "/pending":
                    self._send_json(200, {"pending": receiver._prompt_queue})
                elif self.path.startswith("/result/"):
                    prompt_id = self.path[8:]  # Remove "/result/"
                    result = receiver.get_result(prompt_id)
                    if result is None:
                        self._send_json(202, {"status": "pending", "prompt_id": prompt_id})
                    elif result.get("ready"):
                        image_path = result.get("image_path")
                        if image_path and os.path.exists(image_path):
                            # Return the image file
                            import mimetypes
                            mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
                            self.send_response(200)
                            self.send_header('Content-Type', mime_type)
                            self.send_header('X-Prompt-Id', prompt_id)
                            self.send_header('Access-Control-Allow-Origin', '*')
                            self.end_headers()
                            with open(image_path, 'rb') as f:
                                self.wfile.write(f.read())
                        else:
                            self._send_json(200, {"status": "ready", "prompt_id": prompt_id, "image_path": image_path})
                    else:
                        self._send_json(202, {"status": "processing", "prompt_id": prompt_id})
                else:
                    self._send_json(404, {"error": "Not found"})

        try:
            self._server = HTTPServer(('0.0.0.0', port), RequestHandler)
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
            print(f"[DonutPromptReceiver] Server started on port {port}")
            print(f"[DonutPromptReceiver] POST http://localhost:{port}/prompt")
        except OSError as e:
            if "Address already in use" in str(e):
                print(f"[DonutPromptReceiver] Server already running on port {port}")
            else:
                print(f"[DonutPromptReceiver] Failed to start server: {e}")

    def queue_workflow(self, prompt_text, face_prompt="", negative_prompt=""):
        """Queue the workflow with the new prompt."""
        if not self._workflow or not self._node_id:
            return {"error": "Workflow not registered. Run the workflow once first."}

        try:
            import uuid
            import copy

            # Deep copy and randomize seeds
            workflow = copy.deepcopy(self._workflow)
            new_seed = random.randint(0, 2**32 - 1)
            for node_id, node in workflow.items():
                if isinstance(node, dict) and "inputs" in node:
                    inputs = node["inputs"]
                    if "seed" in inputs and isinstance(inputs["seed"], (int, float)):
                        inputs["seed"] = new_seed
                    if "noise_seed" in inputs and isinstance(inputs["noise_seed"], (int, float)):
                        inputs["noise_seed"] = new_seed

            # Generate prompt_id first so we can inject it
            prompt_id = str(uuid.uuid4())

            # Store prompt text for later retrieval by reporter
            self.store_prompt_text(prompt_id, prompt_text)

            # Mark this node as API-triggered by adding prompts with magic prefix
            # Also include prompt_id so receiver can share with reporter
            if self._node_id and self._node_id in workflow:
                workflow[self._node_id]["inputs"]["_api_prompt"] = f"__API__:{prompt_text}"
                workflow[self._node_id]["inputs"]["_api_prompt_id"] = prompt_id  # Share with reporter
                if face_prompt:
                    workflow[self._node_id]["inputs"]["_api_face_prompt"] = f"__API__:{face_prompt}"
                if negative_prompt:
                    workflow[self._node_id]["inputs"]["_api_negative_prompt"] = f"__API__:{negative_prompt}"

            # Inject prompt_id into reporter node if registered
            if self._reporter_node_id and self._reporter_node_id in workflow:
                workflow[self._reporter_node_id]["inputs"]["_api_prompt_id"] = prompt_id
            payload = {
                "prompt": workflow,
                "prompt_id": prompt_id
            }

            req = urllib.request.Request(
                "http://127.0.0.1:8188/prompt",
                data=json.dumps(payload).encode('utf-8'),
                headers={"Content-Type": "application/json"}
            )

            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                return {"prompt_id": result.get("prompt_id", prompt_id), "status": "queued"}

        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            try:
                return json.loads(error_body)
            except:
                return {"error": error_body}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}


class DonutPromptReceiver:
    """
    HTTP API endpoint for receiving prompts.

    Connect the prompt output to your text input node (e.g., Wildcard Processor).
    Run the workflow once to register it, then POST prompts to the server.

    POST http://localhost:8001/prompt {"prompt": "your text here"}
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "port": ("INT", {"default": 8001, "min": 1024, "max": 65535}),
                "loop_enabled": ("BOOLEAN", {"default": False}),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
                "meta_prompt": ("STRING", {"default": "enhance this prompt for image generation", "multiline": True}),
                "num_prompts": ("INT", {"default": 1, "min": 1, "max": 100}),
                "outgoing_host": ("STRING", {"default": "localhost", "multiline": False}),
                "outgoing_port": ("INT", {"default": 5728, "min": 1, "max": 65535}),
                "outgoing_protocol": (["http", "https"], {"default": "http"}),
                "_api_prompt": ("STRING", {"default": "", "multiline": False}),
                "_api_face_prompt": ("STRING", {"default": "", "multiline": False}),
                "_api_negative_prompt": ("STRING", {"default": "", "multiline": False}),
                "_api_prompt_id": ("STRING", {"default": "", "multiline": False}),
            },
            "hidden": {
                "workflow": "PROMPT",
                "node_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("prompt", "face_prompt", "negative_prompt", "api_info",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "donut"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        import time
        return time.time()

    def run(self, port, loop_enabled=False, user_prompt="", meta_prompt="", num_prompts=1, outgoing_host="localhost", outgoing_port=5728, outgoing_protocol="http", _api_prompt="", _api_face_prompt="", _api_negative_prompt="", _api_prompt_id="", workflow=None, node_id=None):
        receiver = PromptReceiverServer.get_instance()

        # Start server
        receiver.start(port)

        # Check if this is an API-triggered run (has magic prefix)
        if _api_prompt.startswith("__API__:"):
            # API run - extract prompts after magic prefix, don't re-register
            prompt = _api_prompt[8:]  # Remove "__API__:" prefix
            face_prompt = _api_face_prompt[8:] if _api_face_prompt.startswith("__API__:") else ""
            negative_prompt = _api_negative_prompt[8:] if _api_negative_prompt.startswith("__API__:") else ""

            # Store prompt_id for DonutImageReporter to use (fallback if not directly injected)
            if _api_prompt_id:
                receiver._current_prompt_id = _api_prompt_id
                print(f"[DonutPromptReceiver] API run with prompt_id: {_api_prompt_id}")

            print(f"[DonutPromptReceiver] API prompt: '{prompt}'")
            if face_prompt:
                print(f"[DonutPromptReceiver] Face prompt: '{face_prompt}'")
            if negative_prompt:
                print(f"[DonutPromptReceiver] Negative prompt: '{negative_prompt}'")
        else:
            # Manual run - register workflow (has resolved Anything Everywhere connections)
            prompt = ""
            face_prompt = ""
            negative_prompt = ""
            if workflow and node_id:
                receiver.register(workflow, node_id)
                print(f"[DonutPromptReceiver] Workflow registered (manual run)")

            # Only update loop state on manual runs (not API runs which have stale values)
            receiver.set_loop_enabled(loop_enabled)
            if loop_enabled:
                # Store current config so loop can use it
                config = {
                    "user_prompt": user_prompt,
                    "meta_prompt": meta_prompt,
                    "num_prompts": num_prompts,
                    "host": outgoing_host,
                    "port": outgoing_port,
                    "protocol": outgoing_protocol,
                    "incoming_port": port,
                }
                receiver._last_send_config = config

        # API info
        api_info = f"""Incoming API (port {port}):
POST /prompt - Queue prompt(s)
POST /cancel - Cancel all
POST /skip   - Skip current
GET  /health - Status
GET  /result/{{id}} - Get image

{{"prompt": "a cat"}}
{{"prompt": "body", "face_prompt": "face", "negative_prompt": "ugly"}}

Outgoing (Send to Claude):
POST {outgoing_protocol}://{outgoing_host}:{outgoing_port}/prompt"""

        return (prompt, face_prompt, negative_prompt, api_info,)


class DonutImageReporter:
    """
    Reports generated images back to the API.

    Connect this after your final image output. On API-triggered runs,
    it saves the image and makes it available via GET /result/{prompt_id}
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "file_format": (["png", "jpg", "webp", "tiff", "bmp"], {"default": "webp"}),
                "quality": ("INT", {"default": 90, "min": 1, "max": 100, "step": 1}),
                "lossless_webp": (["false", "true"], {"default": "false"}),
                "optimize": (["true", "false"], {"default": "true"}),
            },
            "optional": {
                "_api_prompt_id": ("STRING", {"default": "", "multiline": False}),
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "donut"

    def run(self, images, file_format="webp", quality=90, lossless_webp="false", optimize="true", _api_prompt_id="", node_id=None):
        receiver = PromptReceiverServer.get_instance()

        # Register this node so receiver knows where to inject prompt_id
        if node_id:
            receiver.register_reporter(node_id)

        # Use fallback from receiver if _api_prompt_id wasn't directly injected
        prompt_id = _api_prompt_id or receiver._current_prompt_id

        # If this is an API-triggered run, save the image and store result
        if prompt_id:
            try:
                import folder_paths
                import numpy as np
                from PIL import Image

                # Get temp directory
                output_dir = "/home/user/Programs/ComfyUI-new/ComfyUI/temp/"

                # Save first image (or could save all)
                img_tensor = images[0]
                img_array = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                img = Image.fromarray(img_array)

                # Parse options
                lossless = (lossless_webp == "true")
                opt = (optimize == "true")

                # Save with prompt_id as filename
                filename = f"api_result_{prompt_id}.{file_format}"
                filepath = os.path.join(output_dir, filename)

                if file_format == "png":
                    img.save(filepath, format="PNG", optimize=opt)
                elif file_format == "jpg":
                    img.save(filepath, format="JPEG", quality=quality, optimize=opt)
                elif file_format == "webp":
                    img.save(filepath, format="WEBP", quality=quality, lossless=lossless)
                elif file_format == "tiff":
                    img.save(filepath, format="TIFF", quality=quality)
                elif file_format == "bmp":
                    img.save(filepath, format="BMP")
                else:
                    img.save(filepath)

                # Store result and clear current_prompt_id
                receiver.store_result(prompt_id, filepath)
                receiver._current_prompt_id = None  # Clear after use
                print(f"[DonutImageReporter] Saved result for {prompt_id}")

            except Exception as e:
                print(f"[DonutImageReporter] Error saving result: {e}")
                import traceback
                traceback.print_exc()

        # Pass through the images
        return (images,)


NODE_CLASS_MAPPINGS = {
    "DonutPromptReceiver": DonutPromptReceiver,
    "DonutImageReporter": DonutImageReporter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutPromptReceiver": "Donut Prompt Receiver (HTTP)",
    "DonutImageReporter": "Donut Image Reporter (HTTP)",
}

# Start server on module load
PromptReceiverServer.get_instance().start(8001)
print("[DonutPromptReceiver] Server ready on port 8001")
