"""
Image Steganography using LSB (Least Significant Bit) encoding.
- Embeds a message by scattering bits pseudo-randomly across all pixels (harder to detect)
- Uses a seed for reproducible shuffling (same seed needed to extract)

Usage:
  python steganography.py              -> embed with defaults
  python steganography.py extract      -> extract from default output
  python steganography.py embed  <input> <output> "<message>" [seed]
  python steganography.py extract <image> [seed]
"""

import sys
import os
import random
from PIL import Image

DELIMITER = "<<END>>"

# ── Hardcoded defaults ─────────────────────────────────────────────────────────
DEFAULT_INPUT   = r"C:\Users\houss\Desktop\steganography\Steg-generator\normal.jpeg"
DEFAULT_OUTPUT  = r"C:\Users\houss\Desktop\steganography\Steg-generator\output.png"
DEFAULT_MESSAGE = (
    "hello this is a hidden message embedded using LSB steganography. "
    "The bits are scattered pseudo-randomly across the image to reduce "
    "statistical detectability. Only someone with the correct seed can extract this."
)
DEFAULT_SEED    = 42
# ──────────────────────────────────────────────────────────────────────────────


def text_to_bits(text: str) -> str:
    return "".join(format(byte, "08b") for byte in text.encode("utf-8"))


def embed_message(input_path: str, output_path: str, message: str, seed: int = DEFAULT_SEED) -> None:
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")

    img = Image.open(input_path).convert("RGB")
    pixels = list(img.getdata())

    payload_bits = text_to_bits(message + DELIMITER)
    total_bits   = len(payload_bits)

    # Total available slots = pixels × 3 channels
    total_slots = len(pixels) * 3
    if total_bits > total_slots:
        raise ValueError(
            f"Message too long: needs {total_bits} bits, image supports {total_slots}."
        )

    # Build a shuffled list of (pixel_index, channel_index) slot indices
    rng = random.Random(seed)
    slot_indices = list(range(total_slots))
    rng.shuffle(slot_indices)
    chosen_slots = set(slot_indices[:total_bits])

    # Map slot index → bit value
    slot_to_bit = {slot_indices[i]: int(payload_bits[i]) for i in range(total_bits)}

    # Write bits into pixels
    flat_channels = []
    for pixel in pixels:
        flat_channels.extend(pixel)

    for slot, bit in slot_to_bit.items():
        flat_channels[slot] = (flat_channels[slot] & ~1) | bit

    # Rebuild pixel list
    new_pixels = [
        (flat_channels[i], flat_channels[i+1], flat_channels[i+2])
        for i in range(0, len(flat_channels), 3)
    ]

    out_img = Image.new("RGB", img.size)
    out_img.putdata(new_pixels)
    out_img.save(output_path, format="PNG")

    print(f"✅ Message embedded successfully!")
    print(f"   Input  : {input_path}")
    print(f"   Output : {output_path}")
    print(f"   Seed   : {seed}")
    print(f"   Chars  : {len(message)}")
    print(f"   Bits   : {total_bits} / {total_slots} ({100*total_bits/total_slots:.2f}% capacity)")


def extract_message(image_path: str, seed: int = DEFAULT_SEED) -> str:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    pixels = list(img.getdata())

    flat_channels = []
    for pixel in pixels:
        flat_channels.extend(pixel)

    total_slots = len(flat_channels)

    rng = random.Random(seed)
    slot_indices = list(range(total_slots))
    rng.shuffle(slot_indices)

    bits = ""
    decoded = ""
    for slot in slot_indices:
        bits += str(flat_channels[slot] & 1)
        if len(bits) % 8 == 0:
            char = chr(int(bits[-8:], 2))
            decoded += char
            if decoded.endswith(DELIMITER):
                return decoded[:-len(DELIMITER)]

    raise ValueError("No hidden message found. Wrong seed or image?")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(f"Running with defaults:")
        print(f"  Input  : {DEFAULT_INPUT}")
        print(f"  Output : {DEFAULT_OUTPUT}")
        print(f"  Seed   : {DEFAULT_SEED}")
        print()
        embed_message(DEFAULT_INPUT, DEFAULT_OUTPUT, DEFAULT_MESSAGE, DEFAULT_SEED)
        return

    command = sys.argv[1].lower()

    if command == "embed":
        input_path  = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_INPUT
        output_path = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_OUTPUT
        message     = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_MESSAGE
        seed        = int(sys.argv[5]) if len(sys.argv) > 5 else DEFAULT_SEED
        embed_message(input_path, output_path, message, seed)

    elif command == "extract":
        image_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT
        seed       = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_SEED
        message = extract_message(image_path, seed)
        print(f"🔍 Hidden message: \"{message}\"")

    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()