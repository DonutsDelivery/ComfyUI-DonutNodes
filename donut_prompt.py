"""Bounded, deterministic prompt expansion; no third-party node imports.

Mikey-compatible selection semantics are intentionally retained: file selection
starts at seed % line_count, repeated names accept !/+/-/*, N$$ selects N lines,
filters are whole-word OR matches, and each nesting pass restarts the seed.
"""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path
import json
import random
import re

try:
    from .krea2_variance_integration import enhance_prompt_pair, variance_input_types
except ImportError:
    from krea2_variance_integration import enhance_prompt_pair, variance_input_types

MAX_TEXT = 1_000_000
MAX_PROMPT_SETS = 128
MAX_EXPANSIONS = 10_000
FILE_TOKEN = re.compile(r"(?:(\d+)\$\$)?__([!+*\-]?)([^|\n]*?)(\|[^\n]*?)?__|(?<![\w/*])([A-Za-z][\w/-]*)\*(?![\w*])")
CHOICE = re.compile(r"\{([^{}]*)\}")
NUMBER = re.compile(r"<random:(-?\d*\.?\d+):(-?\d*\.?\d+)>")
MACRO = re.compile(r"%([^%\n]+)%")


def wildcard_roots():
    import folder_paths
    user = Path(folder_paths.get_user_directory()) / "wildcards"
    legacy = Path(folder_paths.__file__).resolve().parent / "wildcards"
    roots = [user, legacy]
    try:
        roots.extend(Path(p) for p in folder_paths.get_folder_paths("wildcards"))
    except KeyError:
        pass
    roots.append(Path(__file__).resolve().parent / "wildcards")
    return tuple(dict.fromkeys(p.resolve() for p in roots))


@lru_cache(maxsize=256)
def _read_lines(path, mtime_ns, size):
    if size > MAX_TEXT:
        raise ValueError(f"Wildcard file exceeds {MAX_TEXT} bytes: {path}")
    # Preserve blank lines/comment-looking lines: the source processor counts them.
    with open(path, encoding="utf-8-sig") as handle:
        return tuple(line.strip() for line in handle)


def directory_fingerprint(roots=None):
    result = []
    for root in (wildcard_roots() if roots is None else roots):
        root = Path(root).resolve()
        for path in sorted(root.rglob("*.txt")) if root.is_dir() else ():
            if path.is_file() and path.resolve().is_relative_to(root):
                stat = path.stat()
                result.append((str(path.resolve()), stat.st_mtime_ns, stat.st_size))
    return tuple(result)


def _file_lines(name, roots):
    name = name.replace("\\", "/")
    if not name or name.startswith("/") or ":" in name or ".." in name.split("/"):
        raise ValueError(f"Unsafe wildcard path: {name!r}")
    for root in roots:
        root = Path(root).resolve()
        path = (root / (name + ".txt")).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Wildcard escapes its directory: {name!r}")
        if path.is_file():
            stat = path.stat()
            lines = _read_lines(str(path), stat.st_mtime_ns, stat.st_size)
            if not lines:
                raise ValueError(f"Empty wildcard file: {path}")
            return lines
    return None


def _macros(text, prompt, extra_pnginfo, now):
    prompt = json.loads(prompt) if isinstance(prompt, str) else (prompt or {})
    extra = json.loads(extra_pnginfo) if isinstance(extra_pnginfo, str) else (extra_pnginfo or {})
    workflow = extra.get("workflow", extra)
    workflow = json.loads(workflow) if isinstance(workflow, str) else workflow
    nodes = list(workflow.get("nodes", [])) if isinstance(workflow, dict) else []
    for graph in workflow.get("definitions", {}).get("subgraphs", []) if isinstance(workflow, dict) else []:
        nodes.extend(graph.get("nodes", []))
    aliases = {}
    for node in nodes:
        for alias in (str(node.get("id")), node.get("title"), node.get("properties", {}).get("Node name for S&R")):
            if alias:
                aliases.setdefault(alias, []).append(node)

    def replace(match):
        key = match[1]
        if key.startswith("date:"):
            values = dict(zip(("yyyy", "yy", "MM", "M", "dd", "d", "hh", "h", "mm", "m", "ss", "s"),
                              (now.strftime("%Y"), now.strftime("%y"), f"{now.month:02}", str(now.month),
                               f"{now.day:02}", str(now.day), f"{now.hour:02}", str(now.hour),
                               f"{now.minute:02}", str(now.minute), f"{now.second:02}", str(now.second))))
            return re.sub("|".join(values), lambda m: values[m[0]], key[5:])
        if "." not in key:
            return match[0]
        alias, widget = key.split(".", 1)
        matches = {str(n["id"]): n for n in aliases.get(alias, [])}
        if not matches and alias in prompt:
            value = prompt[alias].get("inputs", {}).get(widget)
        elif len(matches) == 1:
            uid, node = next(iter(matches.items()))
            value = prompt.get(uid, {}).get("inputs", {}).get(widget)
            if value is None:
                value = node.get("widgets_values_named", {}).get(widget)
        elif len(matches) > 1:
            raise ValueError(f"Ambiguous prompt macro %{key}%; use a node ID")
        else:
            return match[0]
        return str(value) if isinstance(value, (str, int, float, bool)) else match[0]
    return MACRO.sub(replace, text)


def expand_text(text, seed=0, max_depth=128, missing="error", *, roots=None, prompt=None, extra_pnginfo=None):
    if not isinstance(text, str):
        raise TypeError("Prompt must be a string")
    if missing not in ("error", "keep", "empty"):
        raise ValueError("missing must be error, keep, or empty")
    if not 1 <= int(max_depth) <= 1024:
        raise ValueError("max_depth must be between 1 and 1024")
    roots = wildcard_roots() if roots is None else tuple(Path(p) for p in roots)
    seed, seen, expansions = int(seed), set(), 0
    now = datetime.now()
    for _ in range(int(max_depth)):
        if len(text) > MAX_TEXT:
            raise ValueError("Expanded prompt exceeds the text size limit")
        if text in seen:
            raise ValueError("Cyclic wildcard expansion detected")
        seen.add(text)
        old = text
        text = _macros(text, prompt, extra_pnginfo, now)
        rng = random.Random(seed)
        for _choice_depth in range(int(max_depth)):
            if not CHOICE.search(text):
                break
            text, count = CHOICE.subn(lambda m: rng.choice(m[1].split("|")), text)
            expansions += count
            if expansions > MAX_EXPANSIONS:
                raise ValueError("Wildcard expansion budget exceeded")
        if CHOICE.search(text):
            raise ValueError("Nested choices exceed max_depth")
        rng = random.Random(seed)
        text = NUMBER.sub(lambda m: str(round(rng.uniform(float(m[1]), float(m[2])), 4)), text)
        rng = random.Random(seed)
        offset, names, replaced = seed, set(), False

        def replace_file(match):
            nonlocal offset, expansions, replaced
            count, modifier, name, filters, short_name = match.groups()
            name = name or short_name
            count = int(count or 1)
            expansions += max(1, count)
            if count > MAX_EXPANSIONS or expansions > MAX_EXPANSIONS:
                raise ValueError("Wildcard expansion budget exceeded")
            lines = _file_lines(name, roots)
            if lines is None:
                if missing == "error":
                    raise ValueError(f"Wildcard not found: {name!r}; searched {', '.join(map(str, roots))}")
                return match[0] if missing == "keep" else ""
            replaced = True
            current = offset
            if name in names:
                if modifier == "!": current = seed
                elif modifier == "+": current = seed + 1
                elif modifier == "-": current = seed - 1
                else: current = rng.randint(0, 1_000_000)
            selected = []
            words = filters[1:].split("|") if filters else []
            for index in range(count):
                start = (current + index) % len(lines)
                if words:
                    found = next((lines[(start + j) % len(lines)] for j in range(len(lines))
                                  if any(re.search(r"\b" + re.escape(word) + r"\b", lines[(start + j) % len(lines)], re.I)
                                         for word in words)), None)
                    if found is not None:
                        selected.append(found)
                else:
                    selected.append(lines[start])
            names.add(name)
            offset += count
            return ",".join(selected)

        text = FILE_TOKEN.sub(replace_file, text)
        if len(text) > MAX_TEXT:
            raise ValueError("Expanded prompt exceeds the text size limit")
        if text == old:
            if replaced:
                raise ValueError("Cyclic wildcard expansion detected")
            return text
        if not (FILE_TOKEN.search(text) or CHOICE.search(text) or NUMBER.search(text) or MACRO.search(text)):
            return text
    raise ValueError(f"Wildcard expansion exceeds max_depth={max_depth}")


def select_prompt_set(prompt_sets_json, index, fallback):
    """Select the base Prompt or one of its additional variants.

    The connected Prompt card is set 1. JSON rows are additional sets starting
    at set 2, so the active integer has one consistent meaning in the UI.
    """
    if not prompt_sets_json or prompt_sets_json == "[]":
        return dict(fallback), None
    try:
        rows = json.loads(prompt_sets_json)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("Prompt sets must be a JSON array.") from error
    if not isinstance(rows, list) or len(rows) > MAX_PROMPT_SETS:
        raise ValueError(f"Prompt sets must contain between 1 and {MAX_PROMPT_SETS} entries.")
    if not rows:
        return dict(fallback), None
    # The connected Prompt card is position zero; JSON rows follow it. The
    # one-based control wraps safely when incrementing after the last set.
    position = max(0, int(index) - 1) % (len(rows) + 1)
    if position == 0:
        return dict(fallback), 0
    row = rows[position - 1]
    if not isinstance(row, dict):
        raise ValueError(f"Prompt set {position + 1} must be an object.")
    selected = dict(fallback)
    for key in ("face", "scene", "negative"):
        value = row.get(key, selected[key])
        if not isinstance(value, str):
            raise ValueError(f"Prompt set {position + 1} field {key!r} must be text.")
        selected[key] = value
    return selected, position


class DonutText:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**53 - 1, "control_after_generate": True}),
            "max_depth": ("INT", {"default": 128, "min": 1, "max": 1024}),
            "missing": (["error", "keep", "empty"],),
        }, "optional": {"prefix": ("STRING", {"forceInput": True}),
                         "suffix": ("STRING", {"forceInput": True}),
                         "separator": ("STRING", {"default": ""})},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process"
    CATEGORY = "donut/text"

    def process(self, text, seed=0, max_depth=128, missing="error", prefix=None, suffix=None, separator="", prompt=None, extra_pnginfo=None):
        text = separator.join(part for part in (prefix, text, suffix) if part is not None)
        result = expand_text(text, seed, max_depth, missing, prompt=prompt, extra_pnginfo=extra_pnginfo)
        return {"ui": {"text": [result]}, "result": (result,)}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Re-evaluate cheap text only (never CLIP) to include nested date/macros
        # and file content changes in the dependency key, without global RNG.
        parts = [kwargs.get(k) for k in ("prefix", "text", "suffix")]
        text = kwargs.get("separator", "").join(t for t in parts if isinstance(t, str))
        resolved = expand_text(text, kwargs.get("seed", 0) or 0,
                               kwargs.get("max_depth", 128), kwargs.get("missing", "error"),
                               prompt=kwargs.get("prompt"), extra_pnginfo=kwargs.get("extra_pnginfo"))
        return directory_fingerprint(), resolved



class DonutPromptConditioning:
    """One encoding boundary; retain face/full prompts and raw/zeroed negatives."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",), "face": ("STRING", {"forceInput": True}),
            "scene": ("STRING", {"forceInput": True}), "negative": ("STRING", {"forceInput": True}),
            "edit_negative": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
            "separator": ("STRING", {"default": ""}),
            "text_seed": ("INT", {"default": 0, "min": 0, "max": 2**53 - 1}),
        }, "optional": {**variance_input_types(),
            "native_reference_enabled": ("BOOLEAN", {"default": False}),
            "native_reference_a": ("IMAGE",), "native_reference_b": ("IMAGE",),
            "prompt_sets_json": ("STRING", {"default": "[]", "multiline": True, "dynamicPrompts": False}),
            "prompt_set_index": ("INT", {"default": 1, "min": 1, "max": 2**53 - 1, "control_after_generate": True}),
        }}
    RETURN_TYPES = ("STRING", "STRING", "STRING", "CONDITIONING", "CONDITIONING", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("full_text", "face_text", "edit_negative", "positive", "face_positive", "negative_zeroed", "negative_raw")
    FUNCTION = "encode"
    CATEGORY = "donut/text"
    OUTPUT_NODE = True

    def encode(self, clip, face, scene, negative, edit_negative="", separator="", text_seed=0,
               native_reference_enabled=False, native_reference_a=None, native_reference_b=None,
               prompt_sets_json="[]", prompt_set_index=0, **variance_options):
        from nodes import CLIPTextEncode, ConditioningZeroOut
        selected, selected_index = select_prompt_set(
            prompt_sets_json, prompt_set_index,
            {"face": face, "scene": scene, "negative": negative},
        )
        if selected_index and selected_index > 0:
            face, scene, negative = (
                expand_text(selected[key], text_seed)
                for key in ("face", "scene", "negative")
            )
        full = face + separator + scene  # Do not strip/normalise the user's text.
        edit_negative = expand_text(edit_negative, text_seed)
        encoder, cache = CLIPTextEncode(), {}
        references = [image for image in (native_reference_a, native_reference_b) if image is not None] if native_reference_enabled else []
        if native_reference_enabled and not references:
            raise ValueError("Add an image in Reference Guidance, or turn reference guidance off.")
        def encode_once(text, use_references=False):
            key = (text, use_references)
            if key not in cache:
                if use_references:
                    # Krea2 is an optional backend; use its native vision template.
                    from comfy.text_encoders.krea2 import KREA2_TEMPLATE
                    prefix = "".join(f"Reference {chr(65 + index)}: <|vision_start|><|image_pad|><|vision_end|>\n" for index in range(len(references)))
                    tokens = clip.tokenize(prefix + text, images=references, llama_template=KREA2_TEMPLATE)
                    if "qwen3vl_4b" not in tokens:
                        raise ValueError("Reference Guidance requires the Krea2 Qwen3-VL text encoder.")
                    cache[key] = clip.encode_from_tokens_scheduled(tokens)
                else:
                    cache[key] = encoder.encode(clip, text)[0]
            return cache[key]
        positive, face_positive = (encode_once(t, bool(references)) for t in (full, face))
        raw = encode_once(negative)
        positive, face_positive = enhance_prompt_pair(positive, face_positive, **variance_options)
        zeroed = ConditioningZeroOut().zero_out(raw)[0]
        ui = {"text": [full], "donut_final_prompt": [full]}
        if selected_index is not None:
            ui["donut_prompt_set"] = [selected_index + 1]
        return {"ui": ui, "result": (full, face, edit_negative, positive, face_positive, zeroed, raw)}

    @classmethod
    def IS_CHANGED(cls, edit_negative="", text_seed=0, prompt_sets_json="[]", prompt_set_index=0, **kwargs):
        selected, selected_index = select_prompt_set(
            prompt_sets_json, prompt_set_index,
            {"face": kwargs.get("face", ""), "scene": kwargs.get("scene", ""), "negative": kwargs.get("negative", "")},
        )
        expanded = (
            tuple(expand_text(selected[key], text_seed or 0) for key in ("face", "scene", "negative"))
            if selected_index and selected_index > 0 else
            tuple(selected[key] for key in ("face", "scene", "negative"))
        )
        return directory_fingerprint(), selected_index, expanded, expand_text(edit_negative or "", text_seed or 0)


NODE_CLASS_MAPPINGS = {"DonutText": DonutText, "DonutPromptConditioning": DonutPromptConditioning}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutText": "Donut Text · Recursive Wildcards", "DonutPromptConditioning": "Donut Prompt Conditioning"}
