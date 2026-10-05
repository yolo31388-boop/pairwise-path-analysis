"""
游戏内玩家路径分析与热力图系统 - 验收测试
运行方式：python -m pytest tests/test_path_analysis.py -q
共 14 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.path_analysis import PathAnalysis


def walk(system, pid, points, t0=0.0, dt=1.0):
    """辅助：按固定间隔记录一串位置点。"""
    for i, (x, y) in enumerate(points):
        system.record_position(pid, x, y, timestamp=t0 + i * dt)


class TestPathAnalysis:
    def setup_method(self):
        self.system = PathAnalysis()

    # ---- 路径分析：轨迹 + 统计 ----
    def test_case_01(self):
        # 路径分析应记录完整移动轨迹，而不是只有当前位置
        walk(self.system, "p1", [(0, 0), (5, 0), (15, 0), (25, 0)])
        result = self.system.path_analysis("p1")
        traj = result["players"]["p1"]["trajectory"]
        assert len(traj) == 4
        assert [p["x"] for p in traj] == [0.0, 5.0, 15.0, 25.0]
        assert traj[-1]["zone"] == (2, 0)

    def test_case_02(self):
        # 路径统计：区域转移次数与常走路径
        walk(self.system, "p1", [(0, 0), (15, 0), (25, 0)])
        walk(self.system, "p2", [(0, 0), (15, 0), (5, 0)])
        stats = self.system.path_analysis()["statistics"]
        assert stats["transition_counts"][((0, 0), (1, 0))] == 2
        assert stats["transition_counts"][((1, 0), (2, 0))] == 1
        # 最常走的路径排在最前
        assert stats["common_paths"][0] == (((0, 0), (1, 0)), 2)

    def test_case_03(self):
        # 路径统计：移动距离与平均速度
        walk(self.system, "p1", [(0, 0), (3, 4), (6, 8)], dt=2.0)
        p1 = self.system.path_analysis("p1")["players"]["p1"]
        assert p1["distance"] == pytest.approx(10.0)
        assert p1["duration"] == pytest.approx(4.0)
        assert p1["avg_speed"] == pytest.approx(2.5)

    # ---- 热力图：密度 + 时间 ----
    def test_case_04(self):
        # 热力图应反映区域密度：玩家聚集的格子计数更高
        for i in range(5):
            self.system.record_position("p1", 1 + i * 0.1, 1, timestamp=float(i))
        self.system.record_position("p2", 50, 50, timestamp=0.0)
        heat = self.system.heatmap()
        assert heat["cells"][(0, 0)] == 5
        assert heat["cells"][(5, 5)] == 1
        assert heat["max_count"] == 5
        # 密度归一化：最热格子为 1.0，其余按比例
        assert heat["density"][(0, 0)] == pytest.approx(1.0)
        assert heat["density"][(5, 5)] == pytest.approx(0.2)

    def test_case_05(self):
        # 热力图应支持时间维度：时间窗口过滤 + 时间分片
        for t in range(4):
            self.system.record_position("p1", 1, 1, timestamp=float(t))
        for t in range(10, 14):
            self.system.record_position("p1", 55, 55, timestamp=float(t))
        # 时间窗口：只看后半段，热点应转移到 (5,5)
        late = self.system.heatmap(time_window=(10.0, 14.0))
        assert late["cells"] == {(5, 5): 4}
        # 时间分片：热度随时间从 (0,0) 迁移到 (5,5)
        sliced = self.system.heatmap(num_slices=2)
        assert len(sliced["slices"]) == 2
        assert sliced["slices"][0]["cells"] == {(0, 0): 4}
        assert sliced["slices"][1]["cells"] == {(5, 5): 4}

    # ---- 移动模式：聚类 + 分类 ----
    def test_case_06(self):
        # 模式分类：直线通勤 / 静止 / 绕圈巡逻
        walk(self.system, "commuter", [(0, 0), (10, 0), (20, 0), (30, 0)])
        walk(self.system, "camper", [(5, 5), (5, 5), (5, 5), (5, 5)])
        walk(self.system, "patrol", [(5, 5), (15, 5), (15, 15), (5, 5)])
        patterns = self.system.movement_pattern()["patterns"]
        assert patterns["commuter"]["class"] == "direct"
        assert patterns["camper"]["class"] == "stationary"
        assert patterns["patrol"]["class"] == "loop"

    def test_case_07(self):
        # 模式聚类：相似移动路径应被聚到同一簇
        walk(self.system, "a1", [(0, 0), (10, 0), (20, 0), (30, 0)])
        walk(self.system, "a2", [(0, 1), (10, 1), (20, 1), (30, 1)])
        walk(self.system, "b1", [(5, 5), (5, 5), (5, 5), (5, 5)])
        result = self.system.movement_pattern(n_clusters=2)
        patterns = result["patterns"]
        assert patterns["a1"]["cluster"] == patterns["a2"]["cluster"]
        assert patterns["b1"]["cluster"] != patterns["a1"]["cluster"]
        assert len(result["clusters"]) == 2
        members = sorted(m for c in result["clusters"] for m in c["members"])
        assert members == ["a1", "a2", "b1"]

    # ---- 路径预测：模型 + 概率 ----
    def test_case_08(self):
        # 预测模型：根据历史路径预测玩家下一步要去的区域
        for _ in range(3):
            walk(self.system, "p1", [(0, 0), (15, 0), (25, 0)], dt=1.0)
        pred = self.system.path_prediction("p1", steps=2)
        assert pred["current_zone"] == (2, 0)
        # 历史模型：从 (0,0) 出发总是去 (1,0)
        assert pred["model"][(0, 0)] == {(1, 0): 1.0}
        # 回放到 (0,0) 重新开始，预测下一步为 (1,0)
        self.system.record_position("p1", 1, 1, timestamp=100.0)
        pred = self.system.path_prediction("p1", steps=2)
        assert pred["current_zone"] == (0, 0)
        assert pred["steps"][0]["candidates"][0] == ((1, 0), 1.0)
        assert pred["most_likely_path"] == [(0, 0), (1, 0), (2, 0)]

    def test_case_09(self):
        # 预测概率：候选区域必须带概率且概率和为 1
        walk(self.system, "p1", [(0, 0), (15, 0)])
        walk(self.system, "p1", [(0, 0), (15, 0)])
        walk(self.system, "p1", [(0, 0), (0, 15)])
        self.system.record_position("p1", 1, 1, timestamp=100.0)
        pred = self.system.path_prediction("p1", steps=1)
        candidates = dict(pred["steps"][0]["candidates"])
        assert candidates[((1, 0))] == pytest.approx(2 / 3)
        assert candidates[((0, 1))] == pytest.approx(1 / 3)
        assert sum(candidates.values()) == pytest.approx(1.0)

    # ---- 区域分析：停留时间 + 流量 ----
    def test_case_10(self):
        # 区域停留时间：统计玩家在每个区域停留多久
        walk(self.system, "p1", [(1, 1), (2, 1), (3, 1), (15, 1), (16, 1)],
             dt=2.0)
        zones = self.system.zone_analysis()["zones"]
        # 在 (0,0) 停留 3 个间隔 * 2s = 6s，在 (1,0) 停留 1 个间隔 = 2s
        assert zones[(0, 0)]["total_dwell"] == pytest.approx(6.0)
        assert zones[(1, 0)]["total_dwell"] == pytest.approx(2.0)
        assert zones[(0, 0)]["avg_dwell"] == pytest.approx(6.0)

    def test_case_11(self):
        # 区域人流量：访问次数、独立访客与进出流量
        walk(self.system, "p1", [(1, 1), (15, 1), (1, 1)])
        walk(self.system, "p2", [(1, 1), (15, 1)])
        zones = self.system.zone_analysis()["zones"]
        assert zones[(0, 0)]["visits"] == 3
        assert zones[(1, 0)]["visits"] == 2
        assert zones[(0, 0)]["unique_visitors"] == 2
        assert zones[(0, 0)]["flow_out"] == {(1, 0): 2}
        assert zones[(1, 0)]["flow_in"] == {(0, 0): 2}
        assert zones[(0, 0)]["traffic"] > 0

    def test_case_12(self):
        # 区域受欢迎程度：排行榜与最热区域
        for _ in range(3):
            walk(self.system, "p1", [(1, 1), (55, 1), (1, 1)])
        walk(self.system, "p2", [(1, 1), (15, 1)])
        analysis = self.system.zone_analysis()
        assert analysis["most_popular"] == (0, 0)
        assert analysis["ranking"][0] == (0, 0)
        assert set(analysis["ranking"]) == {(0, 0), (1, 0), (5, 0)}

    # ---- 系统行为 ----
    def test_case_13(self):
        # update 推进内部时钟；reset 清空全部轨迹与统计
        self.system.update(1.5)
        self.system.record_position("p1", 0, 0)
        self.system.update(0.5)
        self.system.record_position("p1", 10, 0)
        traj = self.system.path_analysis("p1")["players"]["p1"]["trajectory"]
        assert [p["t"] for p in traj] == [1.5, 2.0]
        self.system.reset()
        result = self.system.path_analysis()
        assert result["players"] == {}
        assert result["statistics"]["total_points"] == 0
        assert self.system.heatmap()["total_points"] == 0

    def test_case_14(self):
        # 端到端：记录 -> 五项分析全部产出有效结果
        walk(self.system, "p1", [(0, 0), (10, 0), (20, 0), (30, 0)])
        walk(self.system, "p2", [(0, 0), (0, 10), (0, 20)])
        assert self.system.path_analysis()["statistics"]["total_points"] == 7
        assert self.system.heatmap()["max_count"] > 0
        assert len(self.system.movement_pattern()["patterns"]) == 2
        pred = self.system.path_prediction("p1", steps=1)
        assert pred["current_zone"] == (3, 0)
        assert len(self.system.zone_analysis()["zones"]) == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
