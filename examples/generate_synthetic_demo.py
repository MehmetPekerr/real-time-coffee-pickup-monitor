"""Generate clearly labelled, non-photorealistic demo evidence.

The generated images are illustrative documentation only. They are not model
predictions, benchmark data, or footage from a real store.
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples" / "synthetic"
WIDTH, HEIGHT = 1280, 720


def text(image, value, origin, scale=0.8, color=(235, 235, 235), thickness=2):
    cv2.putText(
        image,
        value,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def person(image, center, color, person_id, wrist_offset):
    x, y = center
    cv2.circle(image, (x, y - 100), 38, color, -1)
    cv2.rectangle(image, (x - 55, y - 60), (x + 55, y + 100), color, -1)
    cv2.line(image, (x - 35, y + 90), (x - 70, y + 185), color, 18)
    cv2.line(image, (x + 35, y + 90), (x + 70, y + 185), color, 18)
    wrist = (x + wrist_offset[0], y + wrist_offset[1])
    cv2.circle(image, wrist, 13, (255, 255, 255), -1)
    cv2.rectangle(
        image,
        (x - 78, y - 140),
        (x + 78, y + 210),
        (80, 220, 255),
        3,
    )
    text(image, f"track_id={person_id}", (x - 80, y + 245), 0.65, (80, 220, 255))
    return wrist


def scene(event_id, title, event_label, people):
    image = np.full((HEIGHT, WIDTH, 3), (30, 35, 45), dtype=np.uint8)
    cv2.rectangle(image, (40, 100), (1240, 670), (55, 62, 75), -1)

    cv2.rectangle(image, (75, 145), (445, 605), (55, 115, 175), 3)
    cv2.rectangle(image, (835, 145), (1205, 605), (110, 150, 90), 3)
    cv2.rectangle(image, (510, 420), (770, 610), (185, 125, 55), 3)
    text(image, "PERSON TRACKING ROI", (92, 180), 0.7, (110, 210, 255))
    text(image, "RECEIPT READER ROI", (850, 180), 0.7, (160, 230, 160))
    text(image, "COFFEE PICKUP ROI", (530, 455), 0.7, (255, 190, 100))
    cv2.circle(image, (640, 535), 42, (185, 125, 55), 3)
    text(image, "cup", (620, 542), 0.6, (255, 200, 130))

    for data in people:
        wrist = person(image, data["center"], data["color"], data["id"], data["wrist"])
        if data.get("pickup"):
            cv2.line(image, wrist, (640, 535), (255, 255, 255), 3)
            text(image, "wrist interaction", (680, 525), 0.6, (255, 255, 255))

    cv2.rectangle(image, (40, 25), (1240, 80), (15, 18, 25), -1)
    text(image, "SYNTHETIC / ILLUSTRATIVE - NOT REAL CAMERA FOOTAGE", (65, 60), 0.8, (80, 220, 255), 2)
    text(image, f"{event_id}  |  {title}", (65, 125), 0.8, (240, 240, 240), 2)
    cv2.rectangle(image, (850, 625), (1205, 660), (15, 18, 25), -1)
    text(image, f"EVENT: {event_label}", (870, 650), 0.7, (120, 240, 150), 2)
    return image


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scenes = [
        (
            "synthetic_frame_001",
            "Customer pickup",
            "coffee_pickup",
            [
                {
                    "center": (300, 330),
                    "color": (165, 105, 55),
                    "id": 7,
                    "wrist": (300, 190),
                    "pickup": True,
                }
            ],
        ),
        (
            "synthetic_frame_002",
            "Staff interaction filtered",
            "ignored_staff_interaction",
            [
                {
                    "center": (300, 330),
                    "color": (55, 175, 205),
                    "id": 11,
                    "wrist": (300, 190),
                    "pickup": True,
                }
            ],
        ),
        (
            "synthetic_frame_003",
            "Pickup with receipt-reader signal",
            "coffee_pickup",
            [
                {
                    "center": (300, 330),
                    "color": (110, 80, 170),
                    "id": 14,
                    "wrist": (300, 190),
                    "pickup": True,
                }
            ],
        ),
    ]

    for event_id, title, event_label, people in scenes:
        image = scene(event_id, title, event_label, people)
        cv2.imwrite(str(OUTPUT / f"{event_id}.png"), image)

    manifest = {
        "dataset_type": "synthetic_illustrative",
        "warning": "These images are generated geometry, not real footage or model predictions.",
        "files": [f"{event_id}.png" for event_id, _, _, _ in scenes],
    }
    with open(OUTPUT / "manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)


if __name__ == "__main__":
    main()
