"""
游戏内玩家路径分析与热力图系统 - 核心模块

功能：
- 路径分析：记录玩家移动轨迹，统计常走路径（区域转移、距离、速度）
- 热力图：按网格统计区域密度，支持时间窗口与时间分片
- 移动模式：对移动路径聚类，并分类移动模式（静止/直线/巡逻/游荡）
- 路径预测：基于一阶马尔可夫模型预测玩家下一步去向，附概率
- 区域分析：统计每个区域的停留时间与人流量
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import time


@dataclass
class Config:
    pass


Point = Tuple[float, float, float, Tuple[int, int]]  # (t, x, y, zone)


class PathAnalysis:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.zone_size = float(self.config.get("zone_size", 10.0))
        self._time = 0.0
        self._positions: Dict[str, List[Point]] = {}
        self._state = {}
        self._history = []

    # ------------------------------------------------------------------
    # 基础记录
    # ------------------------------------------------------------------
    def update(self, dt: float):
        """推进内部时钟，供未显式给时间戳的采样使用。"""
        self._time += float(dt)

    def reset(self):
        self._time = 0.0
        self._positions = {}
        self._state = {}
        self._history = []

    def record_position(self, player_id: str, x: float, y: float,
                        timestamp: Optional[float] = None,
                        zone: Optional[Tuple[int, int]] = None):
        """记录一次玩家位置采样。"""
        t = self._time if timestamp is None else float(timestamp)
        z = zone if zone is not None else self._zone_of(x, y)
        self._positions.setdefault(player_id, []).append((t, float(x), float(y), z))

    def _zone_of(self, x: float, y: float) -> Tuple[int, int]:
        return (int(math.floor(x / self.zone_size)),
                int(math.floor(y / self.zone_size)))

    def _players(self, player_id: Optional[str]) -> List[str]:
        if player_id is not None:
            return [player_id] if player_id in self._positions else []
        return sorted(self._positions.keys())

    # ------------------------------------------------------------------
    # 路径分析：轨迹 + 统计
    # ------------------------------------------------------------------
    def path_analysis(self, player_id: Optional[str] = None) -> Dict:
        """返回玩家轨迹与路径统计（常走路径、转移次数、距离、速度）。"""
        players = {}
        transition_counts: Dict[Tuple, int] = {}
        total_points = 0
        for pid in self._players(player_id):
            pts = self._positions[pid]
            total_points += len(pts)
            trajectory = [
                {"t": t, "x": x, "y": y, "zone": z} for (t, x, y, z) in pts
            ]
            distance = 0.0
            for i in range(1, len(pts)):
                distance += math.hypot(pts[i][1] - pts[i - 1][1],
                                       pts[i][2] - pts[i - 1][2])
                a, b = pts[i - 1][3], pts[i][3]
                if a != b:
                    transition_counts[(a, b)] = transition_counts.get((a, b), 0) + 1
            duration = pts[-1][0] - pts[0][0] if len(pts) > 1 else 0.0
            players[pid] = {
                "trajectory": trajectory,
                "distance": distance,
                "duration": duration,
                "avg_speed": distance / duration if duration > 0 else 0.0,
                "zones_visited": sorted({p[3] for p in pts}),
            }
        common_paths = sorted(transition_counts.items(),
                              key=lambda kv: (-kv[1], kv[0]))
        return {
            "players": players,
            "statistics": {
                "total_points": total_points,
                "transition_counts": dict(transition_counts),
                "common_paths": common_paths,
            },
        }

    # ------------------------------------------------------------------
    # 热力图：密度 + 时间
    # ------------------------------------------------------------------
    def heatmap(self, grid_size: Optional[float] = None,
                time_window: Optional[Tuple[float, float]] = None,
                num_slices: int = 1) -> Dict:
        """生成热力图。

        - 密度：每个网格的采样计数与归一化密度（0~1）。
        - 时间：可用 time_window 过滤时间段，并用 num_slices 查看热度随时间的变化。
        """
        cell = float(grid_size) if grid_size else self.zone_size
        points = [p for pts in self._positions.values() for p in pts]
        if time_window is not None:
            start, end = time_window
            points = [p for p in points if start <= p[0] <= end]
        else:
            start = min((p[0] for p in points), default=0.0)
            end = max((p[0] for p in points), default=0.0)

        def build_cells(pts):
            cells: Dict[Tuple[int, int], int] = {}
            for (_, x, y, _) in pts:
                key = (int(math.floor(x / cell)), int(math.floor(y / cell)))
                cells[key] = cells.get(key, 0) + 1
            return cells

        cells = build_cells(points)
        max_count = max(cells.values(), default=0)
        density = {k: (v / max_count if max_count else 0.0)
                   for k, v in cells.items()}

        slices = []
        if num_slices > 1 and points:
            span = (end - start) or 1.0
            step = span / num_slices
            for i in range(num_slices):
                s = start + i * step
                e = start + (i + 1) * step
                if i < num_slices - 1:
                    pts = [p for p in points if s <= p[0] < e]
                else:
                    pts = [p for p in points if s <= p[0] <= e]
                slices.append({"start": s, "end": e, "cells": build_cells(pts)})

        return {
            "cell_size": cell,
            "cells": cells,
            "density": density,
            "max_count": max_count,
            "total_points": len(points),
            "time_window": (start, end),
            "slices": slices,
        }

    # ------------------------------------------------------------------
    # 移动模式：聚类 + 分类
    # ------------------------------------------------------------------
    @staticmethod
    def _path_features(pts: List[Point]) -> Dict:
        length = 0.0
        for i in range(1, len(pts)):
            length += math.hypot(pts[i][1] - pts[i - 1][1],
                                 pts[i][2] - pts[i - 1][2])
        displacement = math.hypot(pts[-1][1] - pts[0][1],
                                  pts[-1][2] - pts[0][2]) if len(pts) > 1 else 0.0
        duration = pts[-1][0] - pts[0][0] if len(pts) > 1 else 0.0
        ratio = displacement / length if length > 1e-9 else 0.0
        return {
            "length": length,
            "displacement": displacement,
            "ratio": ratio,
            "duration": duration,
            "avg_speed": length / duration if duration > 0 else 0.0,
        }

    @classmethod
    def _classify(cls, feats: Dict, zone_size: float) -> str:
        if feats["length"] < 1e-6:
            return "stationary"      # 静止
        if feats["ratio"] >= 0.7:
            return "direct"          # 直线通勤
        if feats["ratio"] < 0.3 and feats["displacement"] < 2 * zone_size:
            return "loop"            # 巡逻/绕圈
        return "wandering"           # 游荡探索

    @staticmethod
    def _kmeans(vectors: List[List[float]], k: int, iters: int = 50) -> List[int]:
        """确定性 k-means（纯标准库），返回每个向量的簇编号。"""
        n = len(vectors)
        k = max(1, min(k, n))
        order = sorted(range(n), key=lambda i: vectors[i])
        centroids = [list(vectors[order[int(i * (n - 1) / max(k - 1, 1))]])
                     for i in range(k)]
        assign = [0] * n
        for _ in range(iters):
            changed = False
            for i, v in enumerate(vectors):
                best = min(range(k),
                           key=lambda c: sum((a - b) ** 2
                                             for a, b in zip(v, centroids[c])))
                if best != assign[i]:
                    assign[i] = best
                    changed = True
            for c in range(k):
                members = [vectors[i] for i in range(n) if assign[i] == c]
                if members:
                    centroids[c] = [sum(col) / len(members)
                                    for col in zip(*members)]
            if not changed:
                break
        return assign

    def movement_pattern(self, player_id: Optional[str] = None,
                         n_clusters: int = 3) -> Dict:
        """发现玩家移动规律：对路径特征聚类，并分类移动模式。"""
        pids = [p for p in self._players(player_id)
                if len(self._positions[p]) >= 2]
        patterns = {}
        for pid in pids:
            feats = self._path_features(self._positions[pid])
            patterns[pid] = {
                "features": feats,
                "class": self._classify(feats, self.zone_size),
            }

        clusters = []
        if pids:
            raw = [[math.log1p(f["features"]["length"]),
                    math.log1p(f["features"]["displacement"]),
                    f["features"]["ratio"],
                    f["features"]["avg_speed"]] for f in
                   (patterns[p] for p in pids)]
            scales = [max((v[j] for v in raw), default=0.0) or 1.0
                      for j in range(4)]
            vectors = [[v[j] / scales[j] for j in range(4)] for v in raw]
            labels = self._kmeans(vectors, min(n_clusters, len(pids)))
            for pid, label in zip(pids, labels):
                patterns[pid]["cluster"] = label
            for c in sorted(set(labels)):
                members = [pids[i] for i in range(len(pids))
                           if labels[i] == c]
                centroid = [sum(v[j] for v in
                                (vectors[i] for i in range(len(pids))
                                 if labels[i] == c)) / len(members)
                            for j in range(4)]
                clusters.append({"id": c, "members": members,
                                 "centroid": centroid})
        return {"patterns": patterns, "clusters": clusters}

    # ------------------------------------------------------------------
    # 路径预测：模型 + 概率
    # ------------------------------------------------------------------
    def _transition_model(self, player_id: Optional[str]) -> Dict:
        """一阶马尔可夫转移模型：{zone: {next_zone: probability}}。"""
        counts: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = {}
        for pid in self._players(player_id):
            pts = self._positions[pid]
            for i in range(1, len(pts)):
                a, b = pts[i - 1][3], pts[i][3]
                if a != b:
                    counts.setdefault(a, {})
                    counts[a][b] = counts[a].get(b, 0) + 1
        model = {}
        for a, nxt in counts.items():
            total = sum(nxt.values())
            model[a] = {b: c / total for b, c in nxt.items()}
        return model

    def path_prediction(self, player_id: Optional[str] = None,
                        steps: int = 3) -> Dict:
        """根据历史路径预测玩家接下来要去的区域，并给出概率。"""
        model = self._transition_model(player_id)
        if player_id is not None and self._positions.get(player_id):
            current = self._positions[player_id][-1][3]
        else:
            # 全体玩家：取全局出现次数最多的区域作为起点
            freq: Dict[Tuple[int, int], int] = {}
            for pid in self._players(None):
                for p in self._positions[pid]:
                    freq[p[3]] = freq.get(p[3], 0) + 1
            current = max(freq.items(), key=lambda kv: kv[1])[0] if freq else None

        steps_out = []
        dist = {current: 1.0} if current is not None else {}
        most_likely = [current] if current is not None else []
        for i in range(1, steps + 1):
            nxt_dist: Dict[Tuple[int, int], float] = {}
            for z, p in dist.items():
                for nz, q in model.get(z, {}).items():
                    nxt_dist[nz] = nxt_dist.get(nz, 0.0) + p * q
            candidates = sorted(nxt_dist.items(), key=lambda kv: (-kv[1], kv[0]))
            steps_out.append({"step": i, "candidates": candidates})
            if candidates:
                most_likely.append(candidates[0][0])
            dist = dict(nxt_dist)
        return {
            "current_zone": current,
            "steps": steps_out,
            "most_likely_path": most_likely,
            "model": model,
        }

    # ------------------------------------------------------------------
    # 区域分析：停留时间 + 流量
    # ------------------------------------------------------------------
    def zone_analysis(self) -> Dict:
        """统计每个区域的受欢迎程度、停留时间与人流量。"""
        zones: Dict[Tuple[int, int], Dict] = {}

        def zone_entry(z):
            return zones.setdefault(z, {
                "visits": 0, "unique_visitors": set(),
                "total_dwell": 0.0, "flow_in": {}, "flow_out": {},
            })

        t_min, t_max = None, None
        for pid in self._players(None):
            pts = self._positions[pid]
            for i, (t, x, y, z) in enumerate(pts):
                t_min = t if t_min is None else min(t_min, t)
                t_max = t if t_max is None else max(t_max, t)
                entry = zone_entry(z)
                entry["unique_visitors"].add(pid)
                prev_zone = pts[i - 1][3] if i > 0 else None
                if prev_zone != z:
                    entry["visits"] += 1
                    if prev_zone is not None:
                        prev = zone_entry(prev_zone)
                        prev["flow_out"][z] = prev["flow_out"].get(z, 0) + 1
                        entry["flow_in"][prev_zone] = \
                            entry["flow_in"].get(prev_zone, 0) + 1
                if i + 1 < len(pts):
                    entry["total_dwell"] += max(0.0, pts[i + 1][0] - t)

        span = (t_max - t_min) if (t_min is not None and t_max > t_min) else 0.0
        result = {}
        for z, e in zones.items():
            visits = e["visits"]
            result[z] = {
                "visits": visits,
                "unique_visitors": len(e["unique_visitors"]),
                "total_dwell": e["total_dwell"],
                "avg_dwell": e["total_dwell"] / visits if visits else 0.0,
                "traffic": visits / span if span > 0 else float(visits),
                "flow_in": dict(e["flow_in"]),
                "flow_out": dict(e["flow_out"]),
            }
        ranking = sorted(result.items(),
                         key=lambda kv: (-kv[1]["visits"],
                                         -kv[1]["total_dwell"], kv[0]))
        return {
            "zones": result,
            "ranking": [z for z, _ in ranking],
            "most_popular": ranking[0][0] if ranking else None,
        }
