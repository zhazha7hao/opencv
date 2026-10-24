"""配置文件读写：config.yaml <-> Python 字典"""
from pathlib import Path

import yaml

# 用 __file__ 定位，不管从哪个目录运行都能找到配置文件
CONFIG_PATH = Path(__file__).parent / "config.yaml"

# 配置里缺失某项时使用的默认值
DEFAULTS = {
    "hsv": {"h_min": 100, "h_max": 125, "s_min": 80, "v_min": 60,
            "diff_min": 45},
    "detect": {"area_min": 30, "area_max": 30000,
               "min_width": 3.0, "max_width": 60.0,
               "min_aspect": 1.3, "max_aspect": 20.0,
               "min_fill": 0.25, "max_tilt": 35.0, "core_v": 245},
    "match": {"min_len_ratio": 0.35, "max_angle_diff": 18.0,
              "min_ratio": 0.9, "max_ratio": 3.5, "max_along": 0.6},
}


def load_config(path: Path = CONFIG_PATH) -> dict:
    """读取 config.yaml；文件不存在或缺字段时用 DEFAULTS 自动补齐"""
    cfg = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}      # 空文件返回 None，转成 {}
    else:
        print(f"[警告] 未找到 {path}，使用默认参数")

    for section, values in DEFAULTS.items():
        sec = cfg.setdefault(section, {})      # 缺整个小节 → 补空字典
        for key, default in values.items():
            sec.setdefault(key, default)       # 缺某个键 → 补默认值
    return cfg


def save_config(cfg: dict, path: Path = CONFIG_PATH) -> None:
    """把参数写回 config.yaml"""
    text = f"""# 装甲板识别参数配置
# 用调参工具调好后按 s 一键写入本文件，也可以直接手动修改

hsv:
  h_min: {cfg['hsv']['h_min']}          # 色相下限 (0~179，蓝色约 95~130)
  h_max: {cfg['hsv']['h_max']}          # 色相上限
  s_min: {cfg['hsv']['s_min']}          # 饱和度下限
  v_min: {cfg['hsv']['v_min']}          # 亮度下限
  diff_min: {cfg['hsv']['diff_min']}    # B-R 通道差下限，与 HSV 是“与”关系，双保险抗反光

detect:                   # 灯条筛选（数值按 1080p 像素）
  area_min: {cfg['detect']['area_min']}       # 轮廓最小面积，过小视为噪点
  area_max: {cfg['detect']['area_max']}       # 轮廓最大面积，过大视为整片反光
  min_width: {cfg['detect']['min_width']}     # 灯条短边下限（像素），滤细丝噪点
  max_width: {cfg['detect']['max_width']}     # 灯条短边上限（像素），滤大块光斑/箱体
  min_aspect: {cfg['detect']['min_aspect']}   # 长宽比下限（太低不像灯条）
  max_aspect: {cfg['detect']['max_aspect']}   # 长宽比上限（太高是细丝噪声）
  min_fill: {cfg['detect']['min_fill']}       # 面积 / 外接矩形面积，过滤空洞噪声
  max_tilt: {cfg['detect']['max_tilt']}       # 长轴与竖直方向夹角上限（度），滤横向灯带
  core_v: {cfg['detect']['core_v']}           # 实心灯条的芯部亮度下限（过曝接近 255，反光上不去）

match:                    # 双灯条配对
  min_len_ratio: {cfg['match']['min_len_ratio']}   # 两灯条长度比下限
  max_angle_diff: {cfg['match']['max_angle_diff']} # 两灯条长轴夹角上限（度）
  min_ratio: {cfg['match']['min_ratio']}           # 法向间距 / 平均灯长 下限
  max_ratio: {cfg['match']['max_ratio']}           # 同上上限，防止跨板配对
  max_along: {cfg['match']['max_along']}   # 沿轴错位 / 平均灯长 上限
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[保存] 参数已写入 {path}")
