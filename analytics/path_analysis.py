"""
游戏内玩家路径分析与热力图系统 - 核心模块

功能：
- 路径分析：记录玩家移动轨迹，统计移动距离、速度、常走路径
- 热力图：按网格统计区域密度，支持时间分片观察热度变化
- 移动模式：对玩家轨迹聚类并分类（静止/直线/巡逻/游荡）
- 路径预测：基于马尔可夫转移模型预测玩家下一步去向，带概率
- 区域分析：统计各区域的停留时间与人流量
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import time


@dataclass
class Config:
    grid_size: float = 10.0
    max_history: int = 100000


@dataclass
class Position:
    player_id: str
    x: float
    y: float
    timestamp: float


class PathAnalysis:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.grid_size = float(self.config.get("grid_size", 10.0))
        self._positions: Dict[str, List[Position]] = defaultdict(list)
        self._history: List[Position] = []
        self._state = {}

    # ------------------------------------------------------------------
    # 数据记录
    # ------------------------------------------------------------------
    def record_position(self, player_id: str, x: float, y: float,
                        timestamp: Optional[float] = None) -> None:
        """记录玩家在某个时刻的位置，形成移动轨迹。"""
        ts = time.time() if timestamp is None else float(timestamp)
        pos = Position(str(player_id), float(x), float(y), ts)
        self._positions[pos.player_id].append(pos)
        self._history.append(pos)

    def get_trajectory(self, player_id: str) -> List[Tuple[float, float, float]]:
        """返回玩家完整移动轨迹 [(x, y, timestamp), ...]，按时间排序。"""
        traj = sorted(self._positions.get(str(player_id), []),
                      key=lambda p: p.timestamp)
        return [(p.x, p.y, p.timestamp) for p in traj]

    def update(self, dt: float):
        """按配置清理过期轨迹点（可选）。"""
        max_age = self.config.get("max_age")
        if not max_age:
            return
        cutoff = time.time() - float(max_age)
        for pid in list(self._positions):
            self._positions[pid] = [p for p in self._positions[pid]
                                    if p.timestamp >= cutoff]
            if not self._positions[pid]:
                del self._positions[pid]
        self._history = [p for p in self._history if p.timestamp >= cutoff]

    def reset(self):
        self._state = {}
        self._positions = defaultdict(list)
        self._history = []

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _cell(self, x: float, y: float,
              grid_size: Optional[float] = None) -> Tuple[int, int]:
        gs = grid_size or self.grid_size
        return (int(math.floor(x / gs)), int(math.floor(y / gs)))

    @staticmethod
    def _dist(a: Position, b: Position) -> float:
        return math.hypot(b.x - a.x, b.y - a.y)

    def _sorted_traj(self, player_id: str) -> List[Position]:
        return sorted(self._positions.get(str(player_id), []),
                      key=lambda p: p.timestamp)

    # ------------------------------------------------------------------
    # 路径分析：轨迹 + 统计
    # ------------------------------------------------------------------
    def path_analysis(self, player_id: Optional[str] = None) -> Dict:
        """分析玩家移动路径：轨迹、总距离、平均速度、常走区域与常走转移。"""
        ids = [str(player_id)] if player_id is not None else sorted(self._positions)
        players = {}
        for pid in ids:
            traj = self._sorted_traj(pid)
            total_distance = sum(self._dist(a, b)
                                 for a, b in zip(traj, traj[1:]))
            duration = (traj[-1].timestamp - traj[0].timestamp) if len(traj) > 1 else 0.0
            avg_speed = total_distance / duration if duration > 0 else 0.0
            cells = [self._cell(p.x, p.y) for p in traj]
            cell_counts = Counter(cells)
            transitions = Counter(zip(cells, cells[1:]))
            players[pid] = {
                "trajectory": [(p.x, p.y, p.timestamp) for p in traj],
                "point_count": len(traj),
                "total_distance": total_distance,
                "duration": duration,
                "avg_speed": avg_speed,
                "frequent_cells": cell_counts.most_common(5),
                "common_transitions": [
                    (list(a), list(b), n)
                    for (a, b), n in transitions.most_common(5)
                ],
            }
        return {"player_count": len(players), "players": players}

    # ------------------------------------------------------------------
    # 热力图：密度 + 时间
    # ------------------------------------------------------------------
    def heatmap(self, grid_size: Optional[float] = None,
                start_time: Optional[float] = None,
                end_time: Optional[float] = None,
                time_buckets: int = 4) -> Dict:
        """生成热力图：网格密度、热点区域，以及不同时间段的热度变化。"""
        gs = float(grid_size) if grid_size else self.grid_size
        points = [p for p in self._history
                  if (start_time is None or p.timestamp >= start_time)
                  and (end_time is None or p.timestamp <= end_time)]
        density: Counter = Counter(self._cell(p.x, p.y, gs) for p in points)
        max_density = max(density.values()) if density else 0
        hot_zones = [
            {"cell": list(cell), "count": count,
             "intensity": count / max_density if max_density else 0.0}
            for cell, count in density.most_common(10)
        ]

        time_distribution = []
        if points:
            t0 = min(p.timestamp for p in points)
            t1 = max(p.timestamp for p in points)
            buckets = max(1, int(time_buckets))
            span = (t1 - t0) or 1.0
            bucket_density = [Counter() for _ in range(buckets)]
            for p in points:
                idx = min(int((p.timestamp - t0) / span * buckets), buckets - 1)
                bucket_density[idx][self._cell(p.x, p.y, gs)] += 1
            for i, bc in enumerate(bucket_density):
                top_cell, top_count = (bc.most_common(1)[0]
                                       if bc else (None, 0))
                time_distribution.append({
                    "bucket": i,
                    "start": t0 + span * i / buckets,
                    "end": t0 + span * (i + 1) / buckets,
                    "count": sum(bc.values()),
                    "hot_cell": list(top_cell) if top_cell else None,
                    "hot_count": top_count,
                })

        return {
            "grid_size": gs,
            "total_points": len(points),
            "density": {cell: n for cell, n in density.items()},
            "max_density": max_density,
            "hot_zones": hot_zones,
            "time_distribution": time_distribution,
        }

    # ------------------------------------------------------------------
    # 移动模式：聚类 + 分类
    # ------------------------------------------------------------------
    def _trajectory_features(self, player_id: str) -> Optional[List[float]]:
        traj = self._sorted_traj(player_id)
        if len(traj) < 2:
            return None
        total_distance = sum(self._dist(a, b)
                             for a, b in zip(traj, traj[1:]))
        displacement = self._dist(traj[0], traj[-1])
        duration = traj[-1].timestamp - traj[0].timestamp
        avg_speed = total_distance / duration if duration > 0 else 0.0
        straightness = displacement / total_distance if total_distance > 0 else 0.0
        return [total_distance, displacement, avg_speed, straightness]

    @staticmethod
    def _classify(features: List[float]) -> str:
        total_distance, displacement, avg_speed, straightness = features
        if total_distance < 1e-6 or avg_speed < 1e-6:
            return "stationary"
        if straightness >= 0.9:
            return "direct"
        if straightness < 0.4:
            return "patrol"
        return "wander"

    def movement_pattern(self, n_clusters: int = 3,
                         max_iter: int = 50) -> Dict:
        """对玩家移动轨迹做 k-means 聚类，并对每类移动模式分类。"""
        features = {}
        for pid in sorted(self._positions):
            fv = self._trajectory_features(pid)
            if fv is not None:
                features[pid] = fv
        if not features:
            return {"clusters": {}, "patterns": {},
                    "cluster_patterns": {}, "features": {}}

        ids = sorted(features)
        dim = len(features[ids[0]])
        scales = [max(features[pid][d] for pid in ids) or 1.0
                  for d in range(dim)]
        normed = {pid: [features[pid][d] / scales[d] for d in range(dim)]
                  for pid in ids}

        k = max(1, min(int(n_clusters), len(ids)))
        centroids = [normed[ids[0]][:]]
        while len(centroids) < k:
            farthest, farthest_d = None, -1.0
            for pid in ids:
                d = min(sum((a - b) ** 2 for a, b in zip(normed[pid], c))
                        for c in centroids)
                if d > farthest_d:
                    farthest, farthest_d = pid, d
            centroids.append(normed[farthest][:])

        assignment = {pid: 0 for pid in ids}
        for _ in range(max_iter):
            changed = False
            for pid in ids:
                best = min(range(k), key=lambda c: sum(
                    (a - b) ** 2 for a, b in zip(normed[pid], centroids[c])))
                if best != assignment[pid]:
                    assignment[pid] = best
                    changed = True
            new_centroids = []
            for c in range(k):
                members = [normed[pid] for pid in ids if assignment[pid] == c]
                if members:
                    new_centroids.append([
                        sum(m[d] for m in members) / len(members)
                        for d in range(dim)])
                else:
                    new_centroids.append(centroids[c])
            centroids = new_centroids
            if not changed:
                break

        clusters: Dict[int, List[str]] = defaultdict(list)
        for pid in ids:
            clusters[assignment[pid]].append(pid)

        cluster_patterns = {}
        for c, members in clusters.items():
            centroid_feat = [
                sum(features[pid][d] for pid in members) / len(members)
                for d in range(dim)]
            cluster_patterns[c] = self._classify(centroid_feat)

        patterns = {pid: cluster_patterns[assignment[pid]] for pid in ids}
        return {
            "clusters": {c: sorted(m) for c, m in clusters.items()},
            "patterns": patterns,
            "cluster_patterns": cluster_patterns,
            "features": features,
        }

    # ------------------------------------------------------------------
    # 路径预测：马尔可夫模型 + 概率
    # ------------------------------------------------------------------
    def _build_transition_model(self) -> Dict[Tuple[int, int], Counter]:
        transitions: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
        for pid in self._positions:
            traj = self._sorted_traj(pid)
            cells = [self._cell(p.x, p.y) for p in traj]
            for a, b in zip(cells, cells[1:]):
                transitions[a][b] += 1
        return transitions

    def path_prediction(self, player_id: str, steps: int = 1,
                        top_k: int = 3) -> Dict:
        """根据历史路径的马尔可夫转移模型，预测玩家接下来可能去的区域及概率。"""
        pid = str(player_id)
        traj = self._sorted_traj(pid)
        if not traj:
            return {"player_id": pid, "current_cell": None,
                    "predictions": [], "model": "markov",
                    "transitions_learned": 0}

        transitions = self._build_transition_model()
        current = self._cell(traj[-1].x, traj[-1].y)
        distribution = {current: 1.0}
        for _ in range(max(1, int(steps))):
            nxt: Dict[Tuple[int, int], float] = defaultdict(float)
            for cell, prob in distribution.items():
                outs = transitions.get(cell)
                if not outs:
                    nxt[cell] += prob
                    continue
                total = sum(outs.values())
                for target, count in outs.items():
                    nxt[target] += prob * count / total
            distribution = nxt

        ranked = sorted(distribution.items(), key=lambda kv: (-kv[1], kv[0]))
        predictions = [{"cell": list(cell), "probability": prob}
                       for cell, prob in ranked[:max(1, int(top_k))]]
        return {
            "player_id": pid,
            "current_cell": list(current),
            "steps": steps,
            "predictions": predictions,
            "model": "markov",
            "transitions_learned": sum(len(v) for v in transitions.values()),
        }

    # ------------------------------------------------------------------
    # 区域分析：停留时间 + 流量
    # ------------------------------------------------------------------
    def zone_analysis(self, zones: Dict[str, Tuple[float, float, float, float]]) -> Dict:
        """分析各区域的受欢迎程度：停留时间、访问次数、独立玩家数与人流量。"""
        stats = {}
        for name, (x1, y1, x2, y2) in zones.items():
            stats[name] = {
                "visits": 0,
                "unique_players": set(),
                "total_dwell_time": 0.0,
                "traffic": 0,
            }

        def inside(p: Position, box) -> bool:
            x1, y1, x2, y2 = box
            return x1 <= p.x <= x2 and y1 <= p.y <= y2

        for pid in self._positions:
            traj = self._sorted_traj(pid)
            for name, box in zones.items():
                was_inside = False
                for i, p in enumerate(traj):
                    if inside(p, box):
                        stats[name]["traffic"] += 1
                        stats[name]["unique_players"].add(pid)
                        if i + 1 < len(traj) and inside(traj[i + 1], box):
                            stats[name]["total_dwell_time"] += max(
                                0.0, traj[i + 1].timestamp - p.timestamp)
                        if not was_inside:
                            stats[name]["visits"] += 1
                        was_inside = True
                    else:
                        was_inside = False

        result = {}
        for name, s in stats.items():
            players = sorted(s["unique_players"])
            result[name] = {
                "visits": s["visits"],
                "unique_players": len(players),
                "players": players,
                "total_dwell_time": s["total_dwell_time"],
                "avg_dwell_time": (s["total_dwell_time"] / s["visits"]
                                   if s["visits"] else 0.0),
                "traffic": s["traffic"],
            }
        most_popular = None
        if result:
            most_popular = max(result, key=lambda n: (
                result[n]["traffic"], result[n]["visits"]))
        return {"zones": result, "most_popular": most_popular}
