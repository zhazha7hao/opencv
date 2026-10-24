"""调参工具：拖滑条实时看掩膜/灯条/配对，按 s 把参数写入 config.yaml。

按键：空格=暂停/继续  d=下一帧  a=上一帧  p=打印  s=保存config  r=重载config  q=退出
      [ / ] 切换选中滑条，, / . 微调±1，< / > 微调±10
      v=切到下一个视频，1/2/3=直接跳到第 N 个视频
"""
import copy
from pathlib import Path

import cv2
import numpy as np

from armor_detect import ArmorDetector
from config_io import load_config, save_config

WIN = "tune"

# 当前选中的滑条下标：点中某个滑条即选中它，之后用按键微调
state = {"sel": 0}

# (滑条名, 配置小节, 键, 缩放, 最大值)：浮点参数用缩放存成整数
SLIDERS = [
    ("H_min",          "hsv",    "h_min",         1,   179),
    ("H_max",          "hsv",    "h_max",         1,   179),
    ("S_min",          "hsv",    "s_min",         1,   255),
    ("V_min",          "hsv",    "v_min",         1,   255),
    ("Diff_min",       "hsv",    "diff_min",      1,   255),
    ("Area_min",       "detect", "area_min",      1, 50000),
    ("Area_max",       "detect", "area_max",      1, 500000),
    ("Max_tilt",       "detect", "max_tilt",      1,    90),
    ("Core_v",         "detect", "core_v",        1,   255),
    ("Min_ratio",      "match",  "min_ratio",     10,   30),
    ("Max_ratio",      "match",  "max_ratio",     10,   80),
    ("Max_along",      "match",  "max_along",     100,  100),
    ("Angle_diff",     "match",  "max_angle_diff", 1,   45),
    ("Min_len_ratio",  "match",  "min_len_ratio", 100, 100),
]


def make_cb(i):
    """点中某个滑条就把它设为当前选中项，之后用按键微调"""
    def cb(_value):
        state["sel"] = i
    return cb


def set_trackbars(cfg):
    for name, sec, key, scale, _max in SLIDERS:
        cv2.setTrackbarPos(name, WIN, int(round(cfg[sec][key] * scale)))


def read_params(cfg):
    """读滑条值，组装成和 config.yaml 相同的结构（未调的部分原样带回）"""
    p = copy.deepcopy(cfg)
    for name, sec, key, scale, _max in SLIDERS:
        val = cv2.getTrackbarPos(name, WIN) / scale
        # 整数参数（面积等）必须转回 int，否则 OpenCV 调用会报类型错误
        p[sec][key] = int(round(val)) if isinstance(cfg[sec][key], int) else val
    if p["hsv"]["h_max"] <= p["hsv"]["h_min"]:
        p["hsv"]["h_max"] = min(179, p["hsv"]["h_min"] + 1)
    return p


def build_debug_strip(d):
    """1x3 缩略图：原始掩膜 | 过滤后掩膜 | 通过筛选的灯条，一屏看完每一步"""
    mask = d["mask"]
    filt = mask.copy()
    cv2.drawContours(filt, d["rejected"], -1, 0, -1)
    bars_only = np.zeros_like(mask)
    for g in d["bars"]:
        box = cv2.boxPoints(((g.cx, g.cy), (g.width, g.length), g.theta))
        cv2.fillConvexPoly(bars_only, box.astype(np.int32), 255)

    tiles = []
    for title, img in (("mask 原始掩膜", mask),
                       ("mask_filt 被拒轮廓已涂黑", filt),
                       ("bars 真正送入配对", bars_only)):
        t = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        t = cv2.resize(t, (600, 338))
        cv2.putText(t, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)
        tiles.append(t)
    return np.hstack(tiles)


def draw_overlay(frame, d):
    """主画面：被拒轮廓(橙) / 灯条外接框(蓝)+编号 / 候选连线(黄) / 配对中心(红)"""
    bars = d["bars"]
    vis = frame.copy()
    cv2.drawContours(vis, d["rejected"], -1, (0, 140, 255), 1)

    for k, g in enumerate(bars):
        box = cv2.boxPoints(((g.cx, g.cy), (g.width, g.length), g.theta))
        cv2.polylines(vis, [box.astype(np.int32)], True, (255, 150, 0), 1)
        c = (int(g.cx), int(g.cy))
        cv2.drawMarker(vis, c, (0, 255, 0), cv2.MARKER_CROSS, 16, 1)
        cv2.putText(vis, str(k), (c[0] + 8, c[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for _s, i, j in d["ans"]:
        cv2.line(vis, (int(bars[i].cx), int(bars[i].cy)),
                 (int(bars[j].cx), int(bars[j].cy)), (0, 255, 255), 2)

    for a in d["pairs"]:
        cv2.drawMarker(vis, (int(a[0]), int(a[1])), (0, 0, 255),
                       cv2.MARKER_CROSS, 26, 2)
    return vis


def main():
    cfg = load_config()
    detector = ArmorDetector(cfg)

    # 自动取目录下的第一个输入视频（跳过已生成的 result_*.mp4）
    videos = sorted(p for p in Path(__file__).parent.glob("*.mp4")
                    if not p.name.startswith("result_"))
    if not videos:
        raise RuntimeError("目录下没有输入视频（*.mp4）")
    print("可用视频：" + "  ".join(f"[{i + 1}]{v.name}" for i, v in enumerate(videos)))
    vid_idx = 0
    cap = None
    total = 0

    def open_video(idx):
        """打开第 idx 个视频（自动循环），并回到第一帧暂停状态"""
        nonlocal cap, frame, frame_idx, paused, total, vid_idx
        if cap is not None:
            cap.release()
        vid_idx = idx % len(videos)
        cap = cv2.VideoCapture(str(videos[vid_idx]))
        if not cap.isOpened():
            raise RuntimeError(f"视频打开失败：{videos[vid_idx]}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"视频读不出第一帧：{videos[vid_idx]}")
        frame_idx = 1
        paused = True
        print(f"切换视频：{videos[vid_idx].name}  ({vid_idx + 1}/{len(videos)})")

    open_video(0)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 560, 760)
    for i, (name, _sec, _key, _scale, vmax) in enumerate(SLIDERS):
        cv2.createTrackbar(name, WIN, 0, vmax, make_cb(i))
    set_trackbars(cfg)
    state["sel"] = 0

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("frame", 1280, 720)
    cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("debug", 1820, 380)

    paused = False

    print("按键：空格=暂停/继续  d=下一帧  a=上一帧  p=打印  s=保存config  r=重载config  q=退出")
    print("      点击滑条选中它，[ / ] 切换选中项，, / . 微调±1，< / > 微调±10")
    print("      v=切下一个视频，1/2/3=跳到第 N 个视频")

    while True:
        p = read_params(cfg)
        detector.hsv = p["hsv"]
        detector.mc.update({
            "area_min": p["detect"]["area_min"],
            "area_max": p["detect"]["area_max"],
            "max_tilt": p["detect"]["max_tilt"],
            "core_v": p["detect"]["core_v"],
            "min_len_ratio": p["match"]["min_len_ratio"],
            "max_angle_diff": p["match"]["max_angle_diff"],
            "min_ratio": p["match"]["min_ratio"],
            "max_ratio": p["match"]["max_ratio"],
            "max_along": p["match"]["max_along"],
        })

        d = detector.analyse(frame)
        bars = d["bars"]
        armors = d["pairs"]
        vis = draw_overlay(frame, d)

        sel_name, sel_sec, sel_key, _sc, _mx = SLIDERS[state["sel"]]
        status = (f"frame {frame_idx}/{total}  {'[PAUSED]' if paused else '[PLAY]'}  "
                  f"video={videos[vid_idx].name}  "
                  f"bars={len(bars)} plates={len(armors)} rej={len(d['rejected'])}  "
                  f"area=[{detector.mc['area_min']},{detector.mc['area_max']}]  "
                  f"sel:{sel_name}={p[sel_sec][sel_key]}")
        cv2.putText(vis, status, (14, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)
        cv2.putText(vis, "space/d/a/p/s/r/q/v/1-9  [/]:select  ,.<>:nudge", (14, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.imshow("frame", vis)
        cv2.imshow("debug", build_debug_strip(d))

        key = cv2.waitKey(30 if not paused else 20) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
        elif key in (ord('['), ord(']')):
            # 切换选中的滑条
            step = -1 if key == ord('[') else 1
            state["sel"] = (state["sel"] + step) % len(SLIDERS)
        elif key in (ord(','), ord('.'), ord('<'), ord('>')):
            # 微调选中的滑条：,/. 步长 1，</> 步长 10
            i = state["sel"]
            name, _sec, _key, _scale, vmax = SLIDERS[i]
            step = 10 if key in (ord('<'), ord('>')) else 1
            delta = -step if key in (ord(','), ord('<')) else step
            v = cv2.getTrackbarPos(name, WIN)
            cv2.setTrackbarPos(name, WIN, max(0, min(vmax, v + delta)))
        elif key == ord('v'):
            # 切到下一个视频（循环）
            open_video(vid_idx + 1)
        elif ord('1') <= key <= ord('9'):
            # 直接跳到第 N 个视频（越界则忽略）
            if key - ord('1') < len(videos):
                open_video(key - ord('1'))
        elif key == ord('d'):
            ret, new = cap.read()
            if ret:
                frame = new
                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            paused = True
        elif key == ord('a'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - 2))
            ret, new = cap.read()
            if ret:
                frame = new
                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            paused = True
        elif key == ord('p'):
            print("\n---------- 当前参数 ----------")
            for sec in ("hsv", "detect", "match"):
                print(sec, {k: p[sec][k] for k in p[sec]})

            print("\n---------- 灯条（绿色编号）----------")
            for k, g in enumerate(bars):
                print("  [%d] cx=%7.1f cy=%7.1f  len=%6.1f wid=%5.1f  tilt=%.1f"
                      % (k, g.cx, g.cy, g.length, g.width, g.tilt))

            print("\n---------- 候选配对（贪心挑选顺序，得分越小越先抢）----------")
            for s, i, j in d["ans"]:
                print("  %.4f  %d-%d" % (s, i, j))

            print("\n---------- 被拒配对（含拒绝原因）----------")
            for i, j, why in d["tried"]:
                print("  %d-%d: %s" % (i, j, why))

            print("  已配对中心:", [tuple(round(v, 1) for v in c) for c in armors])
            print("------------------------------\n")
        elif key == ord('s'):
            cfg = p
            save_config(cfg)
        elif key == ord('r'):
            cfg = load_config()
            set_trackbars(cfg)

        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
