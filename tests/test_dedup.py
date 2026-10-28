"""Unit tests for aspect_bucket and perceptual hashing deduplication."""

import pytest
from PIL import Image, ImageDraw

from common.vectors import compute_phash_hex, phash_distance
from ingestion.stage_assets import aspect_bucket


def test_aspect_bucket_standard_ratios():
    # 9:16 vertical video
    name, ratio = aspect_bucket("360x640")
    assert name == "9:16"
    assert ratio == "0.562"

    # 16:9 landscape
    name, ratio = aspect_bucket("1920x1080")
    assert name == "16:9"
    assert ratio == "1.778"

    # 1:1 square
    name, ratio = aspect_bucket("1080x1080")
    assert name == "1:1"
    assert ratio == "1.000"

    # 4:5 social portrait
    name, ratio = aspect_bucket("680x850")
    assert name == "4:5"
    assert ratio == "0.800"

    # 2:3 display
    name, ratio = aspect_bucket("400x600")
    assert name == "2:3"
    assert ratio == "0.667"


def test_aspect_bucket_edge_cases():
    # Invalid size
    assert aspect_bucket("") == ("unknown", "")
    assert aspect_bucket("invalid") == ("unknown", "")
    assert aspect_bucket("0x100") == ("unknown", "")
    assert aspect_bucket("-10x100") == ("unknown", "")


def test_phash_dedup_resized_images():
    # Create base synthetic image
    img1 = Image.new("RGB", (200, 200), color="white")
    draw1 = ImageDraw.Draw(img1)
    draw1.rectangle([20, 20, 100, 100], fill="blue")
    draw1.ellipse([50, 50, 150, 150], fill="red")

    # Resized version of the exact same image
    img2 = img1.resize((100, 100))

    hash1 = compute_phash_hex(img1)
    hash2 = compute_phash_hex(img2)

    # Resized variants should have near-identical perceptual hash (Hamming distance <= 4)
    distance = phash_distance(hash1, hash2)
    assert distance <= 4

    # Completely different image should have high Hamming distance
    img_diff = Image.new("RGB", (200, 200), color="black")
    draw_diff = ImageDraw.Draw(img_diff)
    draw_diff.polygon([(10, 10), (190, 10), (100, 190)], fill="yellow")
    hash_diff = compute_phash_hex(img_diff)

    assert phash_distance(hash1, hash_diff) > 10
