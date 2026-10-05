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


class TestPathAnalysis:
    def setup_method(self):
        self.system = PathAnalysis({"grid_size": 10.0})

    # -- 路径分析：轨迹 ------------------------------------------------
    def test_case_01(self):
        """记录移动轨迹，可按时间顺序回放玩家经过的每个位置。"""
        self.system.record_position("p1", 0, 0, 0.0)
        self.system.record_position("p1", 3, 4, 1.0)
        self.system.record_position("p1", 3, 0, 2.0)

        traj = self.system.get_trajectory("p1")
        assert len(traj) == 3
        assert traj[0] == (0.0, 0.0, 0.0)
        assert traj[1] == (3.0, 4.0, 1.0)
        assert traj[2] == (3.0, 0.0, 2.0)

    # -- 路径分析：统计（距离、时长、速度） ------------------------------
    def test_case_02(self):
        self.system.record_position("p1", 0, 0, 0.0)
        self.system.record_position("p1", 3, 4, 1.0)

        stats = self.system.path_analysis("p1")["players"]["p1"]
        assert stats["point_count"] == 2
        assert stats["total_distance"] == pytest.approx(5.0)
        assert stats["duration"] == pytest.approx(1.0)
        assert stats["avg_speed"] == pytest.approx(5.0)

    # -- 路径分析：常走路径统计 ------------------------------------------
    def test_case_03(self):
        for t in range(3):
            self.system.record_position("p1", 1, 1, float(t))
        self.system.record_position("p1", 11, 1, 3.0)

        stats = self.system.path_analysis("p1")["players"]["p1"]
        assert stats["frequent_cells"][0] == ((0, 0), 3)
        assert ([0, 0], [1, 0], 1) in stats["common_transitions"]

    # -- 热力图：区域密度 -----------------------------------------------
    def test_case_04(self):
        for _ in range(4):
            self.system.record_position("p1", 2, 2, 0.0)
        self.system.record_position("p2", 55, 55, 0.0)

        heat = self.system.heatmap()
        assert heat["density"][(0, 0)] == 4
        assert heat["density"][(5, 5)] == 1
        assert heat["max_density"] == 4
        assert heat["hot_zones"][0]["cell"] == [0, 0]
        assert heat["hot_zones"][0]["intensity"] == pytest.approx(1.0)

    # -- 热力图：时间热度变化 --------------------------------------------
    def test_case_05(self):
        for t in range(3):
            self.system.record_position("p1", 2, 2, float(t))
        self.system.record_position("p1", 55, 55, 10.0)

        heat = self.system.heatmap(time_buckets=4)
        dist = heat["time_distribution"]
        assert len(dist) == 4
        early = sum(b["count"] for b in dist[:1])
        late = dist[-1]["count"]
        assert early == 3
        assert late == 1
        assert dist[0]["hot_cell"] != dist[-1]["hot_cell"]

        recent = self.system.heatmap(start_time=5.0)
        assert recent["total_points"] == 1

    # -- 移动模式：相似路径聚类 ------------------------------------------
    def test_case_06(self):
        for x in (0, 10, 20, 30):
            self.system.record_position("d1", x, 0, float(x))
            self.system.record_position("d2", x, 5, float(x))
        for t in range(3):
            self.system.record_position("s1", 0, 0, float(t))
            self.system.record_position("s2", 50, 50, float(t))

        result = self.system.movement_pattern(n_clusters=2)
        clusters = result["clusters"]

        def cluster_of(pid):
            for cid, members in clusters.items():
                if pid in members:
                    return cid

        assert cluster_of("d1") == cluster_of("d2")
        assert cluster_of("s1") == cluster_of("s2")
        assert cluster_of("d1") != cluster_of("s1")

    # -- 移动模式：分类 -------------------------------------------------
    def test_case_07(self):
        for x in (0, 10, 20, 30):
            self.system.record_position("runner", x, 0, float(x))
        for t in range(3):
            self.system.record_position("guard", 0, 0, float(t))
        self.system.record_position("guard2", 0, 0, 0.0)
        self.system.record_position("guard2", 10, 0, 1.0)
        self.system.record_position("guard2", 0, 0, 2.0)

        patterns = self.system.movement_pattern(n_clusters=3)["patterns"]
        assert patterns["runner"] == "direct"
        assert patterns["guard"] == "stationary"
        assert patterns["guard2"] == "patrol"

    # -- 路径预测：预测去向 ----------------------------------------------
    def test_case_08(self):
        for pid in ("a", "b", "c"):
            self.system.record_position(pid, 0, 0, 0.0)
            self.system.record_position(pid, 10, 0, 1.0)
            self.system.record_position(pid, 20, 0, 2.0)
        self.system.record_position("d", 0, 0, 0.0)
        self.system.record_position("d", 10, 0, 1.0)
        self.system.record_position("d", 10, 10, 2.0)
        self.system.record_position("hero", 0, 0, 0.0)
        self.system.record_position("hero", 10, 0, 1.0)

        result = self.system.path_prediction("hero")
        assert result["model"] == "markov"
        assert result["current_cell"] == [1, 0]
        assert result["predictions"][0]["cell"] == [2, 0]

    # -- 路径预测：概率 -------------------------------------------------
    def test_case_09(self):
        for pid in ("a", "b", "c"):
            self.system.record_position(pid, 0, 0, 0.0)
            self.system.record_position(pid, 10, 0, 1.0)
            self.system.record_position(pid, 20, 0, 2.0)
        self.system.record_position("d", 0, 0, 0.0)
        self.system.record_position("d", 10, 0, 1.0)
        self.system.record_position("d", 10, 10, 2.0)
        self.system.record_position("hero", 0, 0, 0.0)
        self.system.record_position("hero", 10, 0, 1.0)

        preds = self.system.path_prediction("hero")["predictions"]
        assert preds[0]["probability"] == pytest.approx(0.75)
        assert preds[1]["probability"] == pytest.approx(0.25)
        total = sum(p["probability"] for p in preds)
        assert total == pytest.approx(1.0)
        assert all(0.0 < p["probability"] <= 1.0 for p in preds)

    # -- 区域分析：停留时间 ----------------------------------------------
    def test_case_10(self):
        for t in range(5):
            self.system.record_position("p1", 5, 5, float(t))
        self.system.record_position("p1", 25, 25, 5.0)
        self.system.record_position("p1", 25, 25, 6.0)

        zones = {"plaza": (0, 0, 10, 10), "forest": (20, 20, 30, 30)}
        result = self.system.zone_analysis(zones)["zones"]
        assert result["plaza"]["total_dwell_time"] == pytest.approx(4.0)
        assert result["plaza"]["avg_dwell_time"] == pytest.approx(4.0)
        assert result["plaza"]["visits"] == 1
        assert result["forest"]["total_dwell_time"] == pytest.approx(1.0)

    # -- 区域分析：人流量 -----------------------------------------------
    def test_case_11(self):
        for t in range(3):
            self.system.record_position("p1", 5, 5, float(t))
        self.system.record_position("p2", 5, 5, 0.0)

        result = self.system.zone_analysis({"plaza": (0, 0, 10, 10)})["zones"]
        assert result["plaza"]["traffic"] == 4
        assert result["plaza"]["unique_players"] == 2
        assert result["plaza"]["visits"] == 2

    # -- 区域分析：受欢迎程度排名 -----------------------------------------
    def test_case_12(self):
        self.system.record_position("p1", 5, 5, 0.0)
        for t in range(4):
            self.system.record_position("p1", 25, 25, float(t + 1))

        zones = {"plaza": (0, 0, 10, 10), "forest": (20, 20, 30, 30)}
        analysis = self.system.zone_analysis(zones)
        assert analysis["most_popular"] == "forest"

    # -- 重置 ----------------------------------------------------------
    def test_case_13(self):
        self.system.record_position("p1", 0, 0, 0.0)
        self.system.reset()
        assert self.system.get_trajectory("p1") == []
        assert self.system.heatmap()["total_points"] == 0
        assert self.system.path_analysis()["player_count"] == 0

    # -- 多玩家聚合分析 -------------------------------------------------
    def test_case_14(self):
        self.system.record_position("p1", 0, 0, 0.0)
        self.system.record_position("p1", 10, 0, 1.0)
        self.system.record_position("p2", 0, 0, 0.0)
        self.system.record_position("p2", 0, 10, 1.0)

        analysis = self.system.path_analysis()
        assert analysis["player_count"] == 2
        assert set(analysis["players"].keys()) == {"p1", "p2"}

        heat = self.system.heatmap()
        assert heat["density"][(0, 0)] == 2
        assert heat["total_points"] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
