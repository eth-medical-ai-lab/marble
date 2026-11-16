"""
Shared evaluation utilities used across all MARBLE tasks (cube, maze, blox).

Shared image encoding, message formatting, token accounting, and output
helpers used by task-specific evaluators.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PIL import Image


def positive_int(value: str) -> int:
    """Parse a positive CLI budget or worker count before evaluation starts."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative_int(value: str) -> int:
    """Parse a sample limit or starting index, allowing an empty selection."""
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


# ---------------------------------------------------------------------------
# Shared agent model configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """LLM settings shared across all three MARBLE agent environments.

    The shared agent loop now keeps the full conversation history by default,
    so we use a 32k completion budget unless a caller overrides it explicitly.
    """
    model: str = "gpt-4o"
    verbose: bool = False
    local_server_port: Optional[int] = None
    local_server_url: Optional[str] = None
    reasoning_effort: Optional[str] = None
    max_tokens: int = 32768


# ---------------------------------------------------------------------------
# Response pre-processing (thinking blocks, etc.)
# ---------------------------------------------------------------------------

def strip_thinking_blocks(text: str) -> str:
    """Remove ``<think>...</think>`` blocks emitted by reasoning models.

    Call this before any action/answer parsing to prevent the parser from
    picking up stray keywords that appear inside chain-of-thought reasoning.
    Handles multiline content and multiple blocks.
    """
    import re
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def encode_pil_image(pil_image: Image.Image) -> str:
    """Encode a PIL Image to a base64 PNG string."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def image_content_block(b64: str) -> Dict[str, Any]:
    """Return a standard OpenAI-compatible image block."""
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def image_content_block_from_pil(pil_image: Image.Image) -> Dict[str, Any]:
    """Encode a PIL image for a chat observation."""
    return image_content_block(encode_pil_image(pil_image))


# ---------------------------------------------------------------------------
# Message utilities
# ---------------------------------------------------------------------------

def strip_images_for_log(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deep-copy messages, replacing image blocks with ``<|image|>`` placeholder.

    Works for both single and multi-message lists, handling Claude and OpenAI
    image block formats.
    """
    out = []
    for m in messages:
        if m.get("role") != "user":
            out.append(m)
            continue
        content = []
        for blk in m.get("content", []) or []:
            if isinstance(blk, dict) and blk.get("type") in ("image", "image_url"):
                content.append("<|image|>")
            else:
                content.append(blk)
        out.append({"role": m["role"], "content": content})
    return out


# ---------------------------------------------------------------------------
# Variant naming
# ---------------------------------------------------------------------------

def variant_from_flags(
    has_visual: bool,
    has_symbolic: bool,
    has_harness: bool = False,
    extra_suffixes: Optional[List[str]] = None,
) -> str:
    """Build a variant directory name from observation modality flags.

    Examples:
      - visual + symbolic → "visual_symbolic"
      - visual only → "visual_only"
      - visual + harness → "visual_harness"
      - visual + harness + symbolic → "visual_harness_symbolic"

    ``extra_suffixes`` appends additional tags (e.g., ["flat"]).
    """
    if has_visual and has_harness and has_symbolic:
        name = "visual_harness_symbolic"
    elif has_visual and has_harness:
        name = "visual_harness"
    elif has_visual and has_symbolic:
        name = "visual_symbolic"
    elif has_visual:
        name = "visual_only"
    elif has_symbolic:
        name = "symbolic_only"
    else:
        name = "symbolic_only"

    if extra_suffixes:
        for s in extra_suffixes:
            if s:
                name += f"_{s}"

    return name


# ---------------------------------------------------------------------------
# Token usage aggregation & printing
# ---------------------------------------------------------------------------

_TOKEN_KEYS = [
    'prompt_tokens', 'completion_tokens', 'total_tokens',
    'input_tokens', 'output_tokens',
    'reasoning_tokens', 'cache_read_input_tokens',
    'tokens_generated',
]


def aggregate_usage(usage_list: List[Optional[Dict]]) -> Dict[str, int]:
    """Sum token counts across multiple usage dicts.

    Handles both OpenAI (prompt_tokens/completion_tokens) and
    Anthropic (input_tokens/output_tokens) formats.
    Returns only keys that had at least one non-None value.
    """
    totals: Dict[str, int] = {}
    for u in usage_list:
        if not u or not isinstance(u, dict):
            continue
        for key in _TOKEN_KEYS:
            val = u.get(key)
            if isinstance(val, (int, float)):
                totals[key] = totals.get(key, 0) + int(val)
    return totals


def print_run_summary(
    task: str,
    difficulty: str,
    model: str,
    mode: str,
    variant: str,
    successes: int,
    total: int,
    usage: Optional[Dict] = None,
    output_path: str = None,
):
    """Print a boxed run summary with accuracy and token usage to the console."""
    acc = 100 * successes / total if total else 0.0
    iw = 60  # inner width (between the │ borders)

    def _row(text: str) -> str:
        return f"│ {text:<{iw - 2}} │"

    def _kv(key: str, val: str) -> str:
        return _row(f"{key:<12} {val}")

    sep = "─" * iw
    lines = [
        f"┌{sep}┐",
        f"│{'Run Summary':^{iw}}│",
        f"├{sep}┤",
        _kv("Task:", task),
        _kv("Difficulty:", difficulty),
        _kv("Model:", model),
        _kv("Mode:", mode),
        _kv("Variant:", variant),
        f"├{sep}┤",
        _kv("Accuracy:", f"{successes}/{total} ({acc:.1f}%)"),
    ]
    if usage and isinstance(usage, dict):
        input_tok = usage.get('prompt_tokens') or usage.get('input_tokens') or 0
        output_tok = usage.get('completion_tokens') or usage.get('output_tokens') or 0
        reasoning_tok = usage.get('reasoning_tokens') or 0
        total_tok = usage.get('total_tokens') or 0
        if not reasoning_tok and total_tok > input_tok + output_tok:
            reasoning_tok = total_tok - input_tok - output_tok
        tok_str = f"{input_tok:,} in / {output_tok:,} out"
        if reasoning_tok:
            tok_str += f" ({reasoning_tok:,} reasoning)"
        lines.append(_kv("Tokens:", tok_str))
    if output_path:
        max_path = iw - 15
        p = output_path if len(output_path) <= max_path else "..." + output_path[-(max_path - 3):]
        lines.append(f"├{sep}┤")
        lines.append(_kv("Saved to:", p))
    lines.append(f"└{sep}┘")
    print("\n" + "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def create_text_box(text: str, box_width: int = 100, prefix: str = "   ") -> List[str]:
    """Render text inside a Unicode box for console output."""
    lines: List[str] = []
    lines.append(f"{prefix}╔{'═' * (box_width - 3)}╗")
    raw_lines = (text or "").splitlines() or [""]
    for raw in raw_lines:
        if raw == "":
            lines.append(f"{prefix}║ {' ' * (box_width - 6)}  ║")
            continue
        words = raw.split()
        current_line: List[str] = []
        current_length = 0
        for word in words:
            sep = 1 if current_line else 0
            if current_length + len(word) + sep <= box_width - 6:
                current_line.append(word)
                current_length += len(word) + sep
            else:
                line_text = " ".join(current_line)
                lines.append(f"{prefix}║ {line_text}{' ' * (box_width - len(line_text) - 6)}  ║")
                current_line = [word]
                current_length = len(word)
        if current_line:
            line_text = " ".join(current_line)
            lines.append(f"{prefix}║ {line_text}{' ' * (box_width - len(line_text) - 6)}  ║")
    lines.append(f"{prefix}╚{'═' * (box_width - 3)}╝")
    return lines


# ---------------------------------------------------------------------------
# Conversation JSON export
# ---------------------------------------------------------------------------

def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [_json_safe(item) for item in obj]
    if isinstance(obj, dict):
        return {str(key): _json_safe(value) for key, value in obj.items()}
    return str(obj)


def _ext_from_media_type(media_type: str) -> str:
    media_type = (media_type or "").lower()
    if media_type == "image/jpeg":
        return ".jpg"
    if media_type == "image/webp":
        return ".webp"
    return ".png"


def _decode_inline_image_block(block: Dict[str, Any]) -> Optional[tuple[str, bytes]]:
    """Decode an inline base64 image block into media type + raw bytes."""

    block_type = str(block.get("type", ""))
    if block_type == "image_url":
        url = str(block.get("image_url", {}).get("url", ""))
        if url.startswith("data:") and "base64," in url:
            media_type = url.split(":", 1)[1].split(";", 1)[0]
            return media_type, base64.b64decode(url.split("base64,", 1)[1])
        return None

    if block_type == "image":
        src = block.get("source", {})
        if src.get("type") == "base64" and src.get("data"):
            media_type = str(src.get("media_type", "image/png"))
            return media_type, base64.b64decode(src["data"])
        return None

    return None


def _materialize_inline_image_block(
    block: Dict[str, Any],
    save_dir: str,
    message_idx: int,
    image_idx: int,
) -> Optional[tuple[str, str]]:
    """Save one inline image block to disk and return its filename + media type."""

    decoded = _decode_inline_image_block(block)
    if decoded is None:
        return None

    media_type, img_bytes = decoded
    ext = _ext_from_media_type(media_type)
    fname = f"message_{message_idx:03d}_img_{image_idx}{ext}"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, fname), "wb") as f:
        f.write(img_bytes)
    return fname, media_type


def _normalize_content_for_json(content: Any, save_dir: str, msg_idx: int) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return _json_safe(content)

    normalized: List[Dict[str, Any]] = []
    image_idx = 0
    for block in content:
        if not isinstance(block, dict):
            normalized.append({"type": "text", "text": str(block)})
            continue

        block_type = block.get("type", "")
        if block_type == "text":
            normalized.append({"type": "text", "text": str(block.get("text", ""))})
            continue

        if block_type == "image_url":
            url = str(block.get("image_url", {}).get("url", ""))
            saved = _materialize_inline_image_block(block, save_dir, msg_idx, image_idx)
            if saved is not None:
                fname, media_type = saved
                normalized.append({"type": "image", "media_type": media_type, "file": fname})
            else:
                normalized.append({"type": "image_url", "image_url": {"url": url}})
            image_idx += 1
            continue

        if block_type == "image":
            saved = _materialize_inline_image_block(block, save_dir, msg_idx, image_idx)
            if saved is not None:
                fname, media_type = saved
                normalized.append({"type": "image", "media_type": media_type, "file": fname})
            else:
                normalized.append(_json_safe(block))
            image_idx += 1
            continue

        normalized.append(_json_safe(block))

    return normalized


def write_conversation_json(
    messages: List[Dict[str, Any]],
    save_dir: str,
    filename: str = "conversation.json",
) -> None:
    """Write a structured conversation log and record saved image filenames."""

    os.makedirs(save_dir, exist_ok=True)
    payload: Dict[str, Any] = {"messages": []}
    for msg_idx, message in enumerate(messages or []):
        normalized = {
            key: _json_safe(value)
            for key, value in message.items()
            if key != "content"
        }
        normalized["content"] = _normalize_content_for_json(
            message.get("content", ""),
            save_dir,
            msg_idx,
        )
        payload["messages"].append(normalized)

    with open(os.path.join(save_dir, filename), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _conversation_text_image_file(
    block: Dict[str, Any],
    save_dir: str,
    message_idx: int,
    image_idx: int,
) -> Optional[str]:
    saved = _materialize_inline_image_block(block, save_dir, message_idx, image_idx)
    if saved is None:
        return None
    fname, _ = saved
    return fname


def _append_conversation_text_content(
    lines: List[str],
    content: Any,
    save_dir: str,
    message_idx: int,
) -> None:
    if isinstance(content, str):
        for line in content.split("\n"):
            lines.append(f"  {line}")
        return
    if not isinstance(content, list):
        lines.append(f"  {content}")
        return

    image_idx = 0
    for block in content:
        if not isinstance(block, dict):
            lines.append(f"  {block}")
            continue

        block_type = str(block.get("type", ""))
        if block_type == "text":
            for line in str(block.get("text", "")).split("\n"):
                lines.append(f"  {line}")
            continue

        if block_type in ("image_url", "image"):
            fname = _conversation_text_image_file(block, save_dir, message_idx, image_idx)
            if fname is not None:
                lines.append(f"  [IMAGE: {fname}]")
            else:
                lines.append("  [IMAGE]")
            image_idx += 1
            continue

        lines.append(f"  {_json_safe(block)}")


def write_conversation_txt(
    messages: List[Dict[str, Any]],
    save_dir: str,
    filename: str = "conversation.txt",
) -> None:
    """Write a human-readable conversation log for any MARBLE agent run."""

    os.makedirs(save_dir, exist_ok=True)
    sep = "-" * 70
    lines: List[str] = []

    for msg_idx, message in enumerate(messages or []):
        role = str(message.get("role", "message") or "message").lower()
        title = role.upper()

        lines.extend([sep, f"MESSAGE {msg_idx}", sep, "", f"[{title}]"])

        if role == "assistant":
            reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "")
            if reasoning:
                reasoning_tok = max(1, len(reasoning) // 4)
                lines.append(f"  [Reasoning] (... {reasoning_tok} tokens)")

        content = message.get("content", "")
        has_content = bool(content) if isinstance(content, str) else bool(content or [])
        if has_content:
            _append_conversation_text_content(lines, content, save_dir, msg_idx)

        lines.append("")

    with open(os.path.join(save_dir, filename), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

