"""
Live webcam demo for hand gesture recognition.

Usage:
    python scripts/run_webcam.py --mode classical --source 0
    python scripts/run_webcam.py --mode deep --source 0
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import cv2

from src.webcam.streamer import WebcamStreamer
from src.webcam.pipeline import GesturePipeline


def main():
    parser = argparse.ArgumentParser(description="Live gesture recognition demo")
    parser.add_argument("--mode", choices=["classical", "deep"], default="classical",
                        help="Inference mode")
    parser.add_argument("--source", default=0, help="Camera index or video path")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    pipeline = GesturePipeline(mode=args.mode)

    with WebcamStreamer(args.source, args.width, args.height) as stream:
        while True:
            success, frame = stream.read()
            if not success:
                break

            annotated = pipeline.process_frame(frame)
            cv2.imshow("Hand Gesture Recognition", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                pipeline.switch_mode("classical")
                print("Switched to CLASSICAL mode")
            elif key == ord("d"):
                pipeline.switch_mode("deep")
                print("Switched to DEEP mode")
            elif key == ord("s"):
                screenshot_path = Path("docs/results/screenshot.jpg")
                screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(screenshot_path), annotated)
                print(f"Screenshot saved to {screenshot_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
