import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class LightBar:
    cx: float
    cy: float
    theta: float      # 长轴与 x 轴夹角（度，归一化到 [0, 180)）
    tilt: float       # 长轴与竖直方向夹角
    length: float
    width: float
    aspect: float
    area: float


def color_mask(img, hsv_img, hsv):
    b, _, r = cv2.split(img)
    diff = cv2.subtract(b, r)
    _, mask = cv2.threshold(diff, hsv["diff_min"], 255, cv2.THRESH_BINARY)
    gate = cv2.inRange(hsv_img, (hsv["h_min"], hsv["s_min"], hsv["v_min"]),(hsv["h_max"], 255, 255))
    return cv2.bitwise_and(mask, gate)


def bar_axis(cnt):
    rect = cv2.minAreaRect(cnt)
    w, h = rect[1]
    theta = (rect[2] + (90.0 if h > w else 0.0)) % 180.0
    length, width = max(w, h), min(w, h)
    return theta, length, width


def find_bars(mask, hsv_img, mc):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0] if hierarchy is not None else []
    parent_of = [int(h[3]) for h in hierarchy]
    children = {}
    for i, p in enumerate(parent_of):
        if p >= 0:
            children.setdefault(p, []).append(i)

    bars, rejected = [], []
    hole_ok = set()

    def geometry(cnt, area):
        if area < mc["area_min"] or area > mc["area_max"]:
            return None
        theta, length, width = bar_axis(cnt)
        if not (mc["min_width"] <= width <= mc["max_width"]):
            return None

        aspect = length / width
        if not (mc["min_aspect"] <= aspect <= mc["max_aspect"]):
            return None

        fill = area / (length * width)
        if fill < mc["min_fill"]:
            return None

        tilt = abs(theta - 90.0)
        if tilt > mc["max_tilt"]:
            return None

        return theta, length, width, aspect, tilt

    def add_bar(cnt, theta, tilt, length, width, aspect, area):
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            return False
        bars.append(LightBar(M["m10"] / M["m00"], M["m01"] / M["m00"],theta, tilt, length, width, aspect, area))
        return True

    for i, cnt in enumerate(contours):
        if parent_of[i] < 0:
            continue
        area = cv2.contourArea(cnt)
        g = geometry(cnt, area)
        if g is None:
            rejected.append(cnt)
            continue
        theta, length, width, aspect, tilt = g
        x, y, w, h = cv2.boundingRect(cnt)
        if int(hsv_img[y:y + h, x:x + w, 2].max()) < mc["core_v"]:
            rejected.append(cnt)
            continue
        if not add_bar(cnt, theta, tilt, length, width, aspect, area):
            rejected.append(cnt)
            continue
        hole_ok.add(i)

    for i, cnt in enumerate(contours):
        if parent_of[i] >= 0:
            continue
        if any(c in hole_ok for c in children.get(i, [])):
            continue
        area = cv2.contourArea(cnt)
        g = geometry(cnt, area)
        if g is None:
            rejected.append(cnt)
            continue
        theta, length, width, aspect, tilt = g
        x, y, w, h = cv2.boundingRect(cnt)
        if int(hsv_img[y:y + h, x:x + w, 2].max()) < mc["core_v"]:
            rejected.append(cnt)
            continue
        if not add_bar(cnt, theta, tilt, length, width, aspect, area):
            rejected.append(cnt)
            continue

    return bars, rejected




def pick_pairs(bars, mc):
    ans, tried = [], []
    for i in range(len(bars)):
        for j in range(i + 1, len(bars)):
            a, b = bars[i], bars[j]
            d = np.array([b.cx - a.cx, b.cy - a.cy]) #差向量
            dist = float(np.linalg.norm(d)) #距离
            mean_len = (a.length + b.length) / 2 #平均长度
            if mean_len <= 0:
                tried.append((i, j, "mean_len<=0"))
                continue
            ratio = dist / mean_len #距离与平均长度的比
            if ratio > mc["max_ratio"]:
                tried.append((i, j, "总距离比 %.2f > max_ratio %.2f" % (ratio, mc["max_ratio"])))
                continue
            if min(a.length, b.length) / max(a.length, b.length) < mc["min_len_ratio"]:
                tried.append((i, j, "长度比 < min_len_ratio %.2f" % mc["min_len_ratio"]))
                continue
            delta = abs(a.theta - b.theta) #角度差（度）
            if delta > mc["max_angle_diff"]:
                tried.append((i, j, "夹角 %.1f > max_angle_diff %.1f" % (delta, mc["max_angle_diff"])))
                continue

            ang = math.radians((a.theta + b.theta) / 2) #平均方向转弧度，仅用于三角函数
            t = np.array([math.cos(ang), math.sin(ang)]) #轴向量
            n = np.array([-t[1], t[0]]) #法向向量

            dist_t = abs(float(d @ t))          # 沿轴错位
            dist_n = abs(float(d @ n))          # 法向间距

            along = dist_t / mean_len #沿轴错位与平均长度的比
            if along > mc["max_along"]:
                tried.append((i, j, "沿轴错位 %.2f > max_along %.2f" % (along, mc["max_along"])))
                continue
            ratio_n = dist_n / mean_len #法向间距与平均长度的比
            if not (mc["min_ratio"] <= ratio_n <= mc["max_ratio"]):
                tried.append((i, j, "法向间距比 %.2f 不在 [%.2f, %.2f]"
                              % (ratio_n, mc["min_ratio"], mc["max_ratio"])))
                continue

            score = (delta / mc["max_angle_diff"]
                     + along / mc["max_along"]
                     + abs(a.length - b.length) / max(a.length, b.length)
                     + abs(a.width - b.width) / max(a.width, b.width)
                     + abs(ratio_n - (mc["min_ratio"] + mc["max_ratio"]) / 2) / (mc["max_ratio"] - mc["min_ratio"]))
            ans.append((score, i, j))

    ans.sort(key=lambda x: x[0])
    used, pairs = set(), []
    for _, i, j in ans:
        if i in used or j in used:
            continue
        used.update((i, j))
        mx = (bars[i].cx + bars[j].cx) / 2
        my = (bars[i].cy + bars[j].cy) / 2
        pairs.append((mx, my))

    return pairs, ans, tried


class ArmorDetector:
    def __init__(self, cfg):
        self.hsv = cfg["hsv"]
        d = cfg["detect"]
        m = cfg["match"]
        self.mc = {
            "area_min": d["area_min"],
            "area_max": d["area_max"],
            "min_width": d["min_width"],
            "max_width": d["max_width"],
            "min_aspect": d["min_aspect"],
            "max_aspect": d["max_aspect"],
            "min_fill": d["min_fill"],
            "max_tilt": d["max_tilt"],
            "core_v": d["core_v"],
            "min_len_ratio": m["min_len_ratio"],
            "max_angle_diff": m["max_angle_diff"],
            "min_ratio": m["min_ratio"],
            "max_ratio": m["max_ratio"],
            "max_along": m["max_along"],
        }

    def analyse(self, img):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = color_mask(img, hsv_img, self.hsv)
        bars, rejected = find_bars(mask, hsv_img, self.mc)
        pairs, ans, tried = pick_pairs(bars, self.mc)
        return {"mask": mask, "bars": bars, "rejected": rejected,
                "pairs": pairs, "ans": ans, "tried": tried}

    def detect(self, img) -> list[tuple[float, float]]:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = color_mask(img, hsv_img, self.hsv)
        bars, _ = find_bars(mask, hsv_img, self.mc)
        pairs, _, _ = pick_pairs(bars, self.mc)
        return pairs
