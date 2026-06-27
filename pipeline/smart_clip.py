"""Smart Clip Selection — 智能片段选取模块

从用户提供的任意时长视频中，自动识别并选取最适合制作桌面宠物的连贯片段。

核心逻辑：
1. 全视频预采样（1fps）
2. 逐帧质量评估（亮度、清晰度、宠物检测、主体占比）
3. 滑动窗口选取最优连续片段
4. 返回选取结果和评分依据

设计原则：
- 确定性：相同输入始终产生相同输出
- 可解释：每帧有明确的评分维度和数值
- 可干预：用户可以查看、调整选取结果
- 轻量：预分析不依赖重量级模型
"""

import subprocess
import os
import glob
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class FrameScore:
    """单帧质量评分"""
    timestamp: float        # 秒
    frame_path: str         # 帧文件路径
    brightness: float       # 亮度分 0-1（过暗/过曝扣分）
    sharpness: float        # 清晰度分 0-1（Laplacian 方差归一化）
    pet_detected: bool      # 是否检测到宠物
    pet_confidence: float   # 宠物检测置信度 0-1
    pet_area_ratio: float   # 宠物占画面比例 0-1
    composite_score: float  # 综合评分 0-1


@dataclass
class ClipCandidate:
    """候选片段"""
    start_time: float       # 起始时间（秒）
    end_time: float         # 结束时间（秒）
    duration: float         # 时长（秒）
    avg_score: float        # 平均质量分
    min_score: float        # 最低质量分（短板）
    frame_scores: List[FrameScore]  # 该片段内所有帧的评分


@dataclass
class SelectionResult:
    """选取结果"""
    video_path: str
    video_duration: float
    total_frames_analyzed: int
    recommended_clip: ClipCandidate
    all_candidates: List[ClipCandidate]
    selection_reason: str   # 选取理由（面向用户）


def analyze_video(video_path: str, target_duration: float = 8.0,
                  min_duration: float = 6.0, max_duration: float = 12.0,
                  progress_callback=None) -> SelectionResult:
    """分析视频并选取最优片段。

    Args:
        video_path: 视频文件路径
        target_duration: 目标片段时长（秒），默认 8 秒
        min_duration: 最短可接受时长（秒）
        max_duration: 最长可接受时长（秒）
        progress_callback: 进度回调 callback(phase, current, total, detail)

    Returns:
        SelectionResult 包含推荐片段和所有候选片段
    """
    import cv2

    # Step 1: 获取视频信息
    if progress_callback:
        progress_callback("smart_clip", 0, 0, "analyzing_video")

    video_info = _get_video_info(video_path)
    duration = video_info["duration"]

    # 如果视频时长在目标范围内，直接返回整个视频
    if duration <= max_duration:
        return SelectionResult(
            video_path=video_path,
            video_duration=duration,
            total_frames_analyzed=0,
            recommended_clip=ClipCandidate(
                start_time=0, end_time=duration, duration=duration,
                avg_score=1.0, min_score=1.0, frame_scores=[]
            ),
            all_candidates=[],
            selection_reason="视频时长在可处理范围内，使用完整视频"
        )

    # Step 2: 逐秒采样并评分
    if progress_callback:
        progress_callback("smart_clip", 0, int(duration), "sampling")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    _total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # noqa: F841  # kept for diagnostics
    frame_scores: List[FrameScore] = []
    _sample_interval = fps  # noqa: F841  # 每秒取 1 帧(语义保留)

    for sec in range(int(duration)):
        frame_idx = int(sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        score = _evaluate_frame(frame, sec)
        frame_scores.append(score)

        if progress_callback:
            progress_callback("smart_clip", sec + 1, int(duration),
                            f"评分 {sec+1}/{int(duration)}")

    cap.release()

    # Step 3: 滑动窗口选取最优片段
    if progress_callback:
        progress_callback("smart_clip", 0, 0, "selecting")

    target_frames = int(target_duration)
    min_frames = int(min_duration)
    max_frames = int(max_duration)

    best_clip = _sliding_window_select(
        frame_scores, target_frames, min_frames, max_frames
    )

    # Step 4: 生成所有候选片段（用于用户选择）
    all_candidates = _generate_candidates(
        frame_scores, target_frames, min_frames, max_frames, top_n=3
    )

    # Step 5: 生成选取理由
    reason = _generate_reason(best_clip, video_info)

    return SelectionResult(
        video_path=video_path,
        video_duration=duration,
        total_frames_analyzed=len(frame_scores),
        recommended_clip=best_clip,
        all_candidates=all_candidates,
        selection_reason=reason
    )


def _get_video_info(video_path: str) -> dict:
    """获取视频基本信息"""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration,size",
        "-show_entries", "stream=width,height,r_frame_rate,codec_name",
        "-of", "json", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    import json
    data = json.loads(result.stdout)

    fmt = data.get("format", {})
    st = data.get("streams", [{}])[0]

    fps_parts = st.get("r_frame_rate", "30/1").split("/")
    fps = int(fps_parts[0]) / int(fps_parts[1]) if len(fps_parts) == 2 else 30

    return {
        "duration": float(fmt.get("duration", 0)),
        "width": st.get("width", 0),
        "height": st.get("height", 0),
        "fps": fps,
        "codec": st.get("codec_name", ""),
        "size_bytes": int(fmt.get("size", 0)),
    }


def _evaluate_frame(frame, timestamp: float) -> FrameScore:
    """评估单帧质量。

    评分维度：
    1. 亮度：过暗 (<30) 或过曝 (>220) 扣分
    2. 清晰度：Laplacian 方差，越高越清晰
    3. 宠物检测：torchvision Faster R-CNN 检测（如果可用）或简单启发式
    4. 主体占比：前景占画面的比例
    """
    import cv2

    h, w = frame.shape[:2]

    # 亮度评分
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()
    if mean_brightness < 30:
        brightness_score = mean_brightness / 30 * 0.5  # 过暗
    elif mean_brightness > 220:
        brightness_score = (255 - mean_brightness) / 35 * 0.5  # 过曝
    else:
        brightness_score = 1.0 - abs(mean_brightness - 128) / 128 * 0.3

    # 清晰度评分（Laplacian 方差）
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # 归一化：经验值，方差 > 500 算清晰
    sharpness_score = min(1.0, laplacian_var / 500)

    # 宠物检测：优先用 Faster R-CNN，回退到启发式
    pet_detected, pet_confidence, pet_area_ratio = _detect_pet_rcnn(frame)
    if not pet_detected:
        pet_detected, pet_confidence, pet_area_ratio = _detect_pet_heuristic(frame)

    # Pet area scoring: reward medium shots, penalize close-ups
    # Optimal: 10%-50% of frame. Hard reject if >70% (close-up unusable for segmentation)
    if pet_area_ratio > 0.7:
        pet_area_score = 0.0  # hard reject: close-up is unusable
        composite = 0.0  # zero out entire score
    elif pet_area_ratio > 0.5:
        pet_area_score = 0.4  # penalty: slightly close
        composite = (
            brightness_score * 0.15 +
            sharpness_score * 0.2 +
            pet_confidence * 0.25 +
            pet_area_score * 0.4
        )
    elif pet_area_ratio >= 0.1:
        pet_area_score = 1.0  # optimal range
        composite = (
            brightness_score * 0.2 +
            sharpness_score * 0.3 +
            pet_confidence * 0.3 +
            pet_area_score * 0.2
        )
    elif pet_area_ratio >= 0.05:
        pet_area_score = 0.6
        composite = (
            brightness_score * 0.2 +
            sharpness_score * 0.3 +
            pet_confidence * 0.3 +
            pet_area_score * 0.2
        )
    else:
        pet_area_score = 0.2
        composite = (
            brightness_score * 0.2 +
            sharpness_score * 0.3 +
            pet_confidence * 0.3 +
            pet_area_score * 0.2
        )

    return FrameScore(
        timestamp=timestamp,
        frame_path="",  # 不保存帧文件，只做评分
        brightness=brightness_score,
        sharpness=sharpness_score,
        pet_detected=pet_detected,
        pet_confidence=pet_confidence,
        pet_area_ratio=pet_area_ratio,
        composite_score=composite
    )


def _detect_pet_rcnn(frame) -> Tuple[bool, float, float]:
    """Detect pet using torchvision Faster R-CNN (if available).

    Falls back to heuristic if detection fails.
    """
    try:
        from pipeline.pet_detector import detect_pet_bbox
        bbox = detect_pet_bbox(frame)
        if bbox is not None:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1) / (h * w)
            # Confidence not available from bbox-only API; use 0.9 as
            # a signal that detection succeeded (threshold already passed).
            return True, 0.9, area
    except Exception:
        pass
    return False, 0.0, 0.0


def _detect_pet_heuristic(frame) -> Tuple[bool, float, float]:
    """简单启发式宠物检测。

    当前版本用图像特征估算（已由 Faster R-CNN 替代 YOLO）：
    1. 检测显著性区域（与背景对比度高的区域）
    2. 计算该区域的面积比例
    3. 判断是否可能是宠物（面积 5%-60%，非边缘位置）
    """
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 简单阈值分割找前景
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 找最大连通域
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, 0.0, 0.0

    largest = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(largest) / (h * w)

    # 计算前景与背景的对比度
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    fg_mean = gray[mask > 0].mean() if mask.sum() > 0 else 0
    bg_mean = gray[mask == 0].mean() if (mask == 0).sum() > 0 else 128
    contrast = abs(fg_mean - bg_mean) / 255

    # 判断是否可能是宠物
    # 宠物通常占画面 5%-60%，且与背景有一定对比度
    if 0.05 < area_ratio < 0.6 and contrast > 0.1:
        confidence = min(1.0, contrast * 2 + (0.3 - abs(area_ratio - 0.3)) * 2)
        return True, confidence, area_ratio
    else:
        return False, contrast * 0.5, area_ratio


def _sliding_window_select(scores: List[FrameScore], target: int,
                           min_len: int, max_len: int) -> ClipCandidate:
    """滑动窗口选取最优连续片段。

    策略：
    1. 以 target 长度的窗口滑动
    2. 计算窗口内帧的平均综合分
    3. 额外惩罚：窗口内最低分过低（避免包含坏帧）
    4. 选择得分最高的窗口
    """
    if len(scores) <= target:
        return ClipCandidate(
            start_time=0,
            end_time=scores[-1].timestamp + 1 if scores else 0,
            duration=len(scores),
            avg_score=sum(s.composite_score for s in scores) / max(len(scores), 1),
            min_score=min((s.composite_score for s in scores), default=0),
            frame_scores=scores
        )

    best_score = -1
    best_start = 0
    best_len = target

    for length in range(min_len, min(max_len + 1, len(scores) + 1)):
        for start in range(len(scores) - length + 1):
            window = scores[start:start + length]
            avg = sum(s.composite_score for s in window) / length
            min_s = min(s.composite_score for s in window)

            # 综合得分：平均分 × 0.7 + 最低分 × 0.3（惩罚短板）
            score = avg * 0.7 + min_s * 0.3

            if score > best_score:
                best_score = score
                best_start = start
                best_len = length

    selected = scores[best_start:best_start + best_len]
    return ClipCandidate(
        start_time=selected[0].timestamp,
        end_time=selected[-1].timestamp + 1,
        duration=best_len,
        avg_score=sum(s.composite_score for s in selected) / best_len,
        min_score=min(s.composite_score for s in selected),
        frame_scores=selected
    )


def _generate_candidates(scores: List[FrameScore], target: int,
                         min_len: int, max_len: int,
                         top_n: int = 3) -> List[ClipCandidate]:
    """生成 top-N 候选片段，供用户选择。"""
    candidates = []
    remaining = scores.copy()

    for _ in range(top_n):
        if len(remaining) < min_len:
            break

        clip = _sliding_window_select(remaining, target, min_len, max_len)
        candidates.append(clip)

        # 移除已选区域，避免重叠
        overlap_start = max(0, int(clip.start_time))
        overlap_end = min(len(remaining), int(clip.end_time))
        remaining = remaining[:overlap_start] + remaining[overlap_end:]

    return candidates


def _generate_reason(clip: ClipCandidate, video_info: dict) -> str:
    """生成面向用户的选取理由。"""
    reasons = []

    if clip.avg_score > 0.8:
        reasons.append("整体质量优秀")
    elif clip.avg_score > 0.6:
        reasons.append("整体质量良好")
    else:
        reasons.append("当前为可选范围内的最佳片段")

    if clip.min_score < 0.4:
        reasons.append("部分帧光线或清晰度稍弱")

    duration = video_info["duration"]
    if duration > clip.duration * 2:
        reasons.append(f"原始视频 {duration:.0f} 秒，已选取最优 {clip.duration:.0f} 秒")

    return "；".join(reasons)


def extract_selected_clip(video_path: str, clip: ClipCandidate,
                          output_path: str, fps: int = 10) -> List[str]:
    """提取选定片段的帧。

    Args:
        video_path: 原始视频路径
        clip: 选定的片段
        output_path: 输出帧目录
        fps: 提取帧率

    Returns:
        提取的帧文件路径列表
    """
    os.makedirs(output_path, exist_ok=True)

    duration = clip.end_time - clip.start_time
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(clip.start_time),
        "-i", video_path,
        "-t", str(duration),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        os.path.join(output_path, "frame_%04d.jpg")
    ]
    subprocess.run(cmd, capture_output=True)

    frames = sorted(glob.glob(os.path.join(output_path, "frame_*.jpg")))
    return frames
