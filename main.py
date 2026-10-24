import cv2

from armor_detect import ArmorDetector
from config_io import load_config

def process(video, cfg):
    detector = ArmorDetector(cfg)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频：{video}")
    Width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out_path = f"result_{video}"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),fps, (Width, Height))
    if not writer.isOpened():
        raise SystemExit(f"无法创建输出视频：{out_path}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        armors = detector.detect(frame)
        for armor in armors:
            cv2.circle(frame, (int(armor[0]), int(armor[1])), 7, (0, 0, 255), -1)
        writer.write(frame)
    cap.release()
    writer.release()


def main():
    cfg = load_config()
    process("1.mp4", cfg)
    process("2.mp4", cfg)

if __name__ == "__main__":
    main()
