"""mesh-uv-unwrap 的验收测试：10 个缺陷各 1 条 + 5 条组合/边界。

全部用例都直接调用 `unwrap.atlas` 的桩函数，因此初始状态下 15 条全红，且没有
collection / import / syntax error。测试只依赖标准库 math 与 pytest。
"""

import math

import pytest

from unwrap.atlas import (
    UnwrapError,
    normalize_island,
    pack_islands,
    project_island,
    rotate_to_fit,
    split_islands,
    texel_density,
    unwrap_batch,
    validate_uv,
)

EPS = 1e-6


# --------------------------------------------------------------------------- #
# 几何辅助：只构造网格数据，不参与断言实现
# --------------------------------------------------------------------------- #


def mesh(vertices, faces):
    return {"vertices": [list(v) for v in vertices], "faces": [list(f) for f in faces]}


def quad_xy(size=1.0):
    """XY 平面上一个正方形（两个三角形，法线 +Z）。"""
    s = float(size)
    return mesh(
        [[0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0]],
        [[0, 1, 2], [0, 2, 3]],
    )


def quad_xz():
    """XZ 平面上的正方形（两个三角形，法线 -Y），用于检验主轴选择。"""
    return mesh(
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        [[0, 1, 2], [0, 2, 3]],
    )


def folded_90_mesh():
    """两个面共享一条边，法线夹角正好 90 度：+Z 面与 -Y 面。"""
    return mesh(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 1, 2], [0, 1, 3]],
    )


def tri_area_2d(a, b, c):
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2.0


def tri_area_3d(a, b, c):
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    cx, cy, cz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    return math.sqrt(cx * cx + cy * cy + cz * cz) / 2.0


def polygon_area(pts):
    """鞋带公式，正负号无关。"""
    acc = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        acc += x1 * y2 - x2 * y1
    return abs(acc) / 2.0


def bbox(pts):
    us = [p[0] for p in pts]
    vs = [p[1] for p in pts]
    return min(us), min(vs), max(us), max(vs)


def rects_apart(r1, r2, padding):
    """两个矩形在每个轴上都至少相距 padding（外扩盒不相交）。"""
    def gap(r, s):
        # 一维区间 [r0, r1] 与 [s0, s1] 的间隔
        return max(s[0] - r[1], r[0] - s[1])

    return gap((r1[0], r1[2]), (r2[0], r2[2])) >= padding - EPS or gap(
        (r1[1], r1[3]), (r2[1], r2[3])
    ) >= padding - EPS


# --------------------------------------------------------------------------- #
# 缺陷 1：阈值用弧度比较度数（60 度被当成 60 弧度，于是永不切分）
# --------------------------------------------------------------------------- #


def test_split_islands_angle_threshold_is_in_degrees():
    m = folded_90_mesh()
    # 两面夹角 90 度 > 阈值 60 度 -> 必须在公共边切开，得到 2 个岛
    islands = split_islands(m, 60.0)
    assert len(islands) == 2
    assert sorted(sorted(island) for island in islands) == [[0], [1]]


# --------------------------------------------------------------------------- #
# 缺陷 2：投影不按主轴（斜面上纹理被拉伸）
# --------------------------------------------------------------------------- #


def test_project_island_uses_dominant_axis():
    m = quad_xz()  # 整个岛的法线是 -Y，主轴是 y 而不是 z
    out = project_island(m, [0, 1], "-y")
    assert len(out["uv"]) == 4
    # 正交投影且选对主轴时，2D 面积必须等于 3D 面积
    assert out["area2d"] == pytest.approx(out["area3d"], abs=EPS)
    assert out["area3d"] == pytest.approx(1.0, abs=EPS)


# --------------------------------------------------------------------------- #
# 缺陷 3：归一化不保持长宽比（圆变椭圆）
# --------------------------------------------------------------------------- #


def test_normalize_island_keeps_aspect_ratio():
    uv = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]  # 2:1 的长方形
    out = normalize_island(uv)
    assert len(out) == len(uv)
    u0, v0, u1, v1 = bbox(out)
    assert u0 >= -EPS and v0 >= -EPS and u1 <= 1 + EPS and v1 <= 1 + EPS
    assert (u1 - u0) == pytest.approx((v1 - v0) * 2.0, abs=EPS)


# --------------------------------------------------------------------------- #
# 缺陷 4：打包不留 padding（相邻岛采样串色）
# --------------------------------------------------------------------------- #


def test_pack_islands_reserves_padding():
    island = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    islands = [list(island), list(island)]
    padding = 0.05
    out = pack_islands(islands, padding, 4.0)
    assert out["overflow"] == []
    uv = out["uv"]
    assert len(uv) == 8
    r1 = bbox(uv[:4])
    r2 = bbox(uv[4:])
    assert rects_apart(r1, r2, padding)
    # 也必须留出到图集边界的 padding
    for r in (r1, r2):
        assert r[0] >= padding - EPS
        assert r[1] >= padding - EPS
        assert r[2] <= 1.0 - padding + EPS
        assert r[3] <= 1.0 - padding + EPS


# --------------------------------------------------------------------------- #
# 缺陷 5：旋转只试 0 度（长条岛浪费空间）
# --------------------------------------------------------------------------- #


def test_rotate_to_fit_tries_quarter_turns():
    strip = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.1), (0.0, 0.1)]  # 长条，宽 1 高 0.1
    box = (0.2, 1.0)  # 0 度放不进；转 90/270 度后 0.1 x 1.0 才放得进
    out = rotate_to_fit(strip, box)
    assert out["rotation"] in (90, 270)
    w, h = out["bbox"]
    assert w <= box[0] + EPS and h <= box[1] + EPS
    assert sorted([round(w, 9), round(h, 9)]) == sorted([0.1, 1.0])


# --------------------------------------------------------------------------- #
# 缺陷 6：纹素密度不归一（不同岛密度差 3 倍）
# --------------------------------------------------------------------------- #


def test_texel_density_is_normalized_across_islands():
    # 大岛 2x2（3D 面积 4），小岛 1x1（3D 面积 1）；两个岛各自铺满 0..1
    m = mesh(
        [[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0],
         [3, 0, 0], [4, 0, 0], [4, 1, 0], [3, 1, 0]],
        [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]],
    )
    unit = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    uv = unit + unit  # 每岛都铺满，uv 面积 1+1，3D 面积 4+1
    density = texel_density(uv, m)
    # 归一化后：总 UV 面积 / 总 3D 面积；未归一化会得到 1/4 或 1/1（差 4 倍）
    assert density == pytest.approx(2.0 / 5.0, abs=EPS)


# --------------------------------------------------------------------------- #
# 缺陷 7：UV 校验不查重叠（重叠区互相覆盖）
# --------------------------------------------------------------------------- #


def test_validate_uv_detects_overlap():
    tri = [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5)]
    shifted = [(0.1, 0.1), (0.6, 0.1), (0.1, 0.6)]  # 与上一个三角形面积重叠
    issues = validate_uv(tri + shifted)
    assert issues, "重叠的 UV 必须被报出来"
    assert any("重叠" in s for s in issues)
    # 完全不重叠的合法 UV 不应报问题
    clean = [(0.0, 0.0), (0.4, 0.0), (0.0, 0.4), (0.5, 0.5), (0.9, 0.5), (0.5, 0.9)]
    assert validate_uv(clean) == []


# --------------------------------------------------------------------------- #
# 缺陷 8：退化三角形不跳过（UV 出现 NaN）
# --------------------------------------------------------------------------- #


def test_validate_uv_flags_degenerate_triangle():
    collinear = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]  # 三点共线，面积 0
    issues = validate_uv(collinear)
    assert issues, "退化三角形必须被报出来，而不是算出 NaN"
    assert any("退化" in s for s in issues)


# --------------------------------------------------------------------------- #
# 缺陷 9：批量共用 UV 缓冲（网格之间串）
# --------------------------------------------------------------------------- #


def test_unwrap_batch_isolates_uv_per_mesh():
    cfg = {"angle_threshold": 60.0, "padding": 0.02, "size": 1024, "axis": None}
    meshes = [quad_xy(1.0), quad_xy(1.0)]
    out = unwrap_batch(meshes, cfg)
    results = out["results"]
    assert len(results) == 2
    a, b = results[0]["uv"], results[1]["uv"]
    assert a is not b, "不同网格不能共用同一个 UV 缓冲"
    assert a == b, "同样的网格应得到同样的 UV 值"
    a[0] = (0.123, 0.456)
    assert b[0] != (0.123, 0.456), "改一个网格的 UV 不能影响另一个"


# --------------------------------------------------------------------------- #
# 缺陷 10：溢出岛被静默丢弃（最终 UV 少一块）
# --------------------------------------------------------------------------- #


def test_pack_islands_reports_overflow_instead_of_dropping():
    big = [(0.0, 0.0), (0.9, 0.0), (0.9, 0.9), (0.0, 0.9)]
    islands = [list(big), list(big)]  # 两个 0.9x0.9 的岛塞不进同一个 size
    out = pack_islands(islands, 0.05, 4.0)
    assert out["overflow"] == [1], "放不下的岛必须显式登记下标"
    # 溢出的岛被跳过，但非溢出的岛一个点都不能少
    assert len(out["uv"]) == 4


# --------------------------------------------------------------------------- #
# 组合 / 边界 1：空网格返回空
# --------------------------------------------------------------------------- #


def test_empty_mesh_returns_empty_result():
    empty = mesh([], [])
    assert split_islands(empty, 60.0) == []
    assert unwrap_batch([empty], {"angle_threshold": 60.0, "padding": 0.0, "size": 256})[
        "results"
    ][0]["uv"] == []


# --------------------------------------------------------------------------- #
# 组合 / 边界 2：padding < 0 抛 UnwrapError
# --------------------------------------------------------------------------- #


def test_pack_islands_negative_padding_raises():
    unit = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    with pytest.raises(UnwrapError):
        pack_islands([unit], -0.01, 1024)


# --------------------------------------------------------------------------- #
# 组合 / 边界 3：单面网格的 UV 全部落在 0..1
# --------------------------------------------------------------------------- #


def test_single_face_uv_stays_inside_unit_square():
    m = mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    cfg = {"angle_threshold": 60.0, "padding": 0.01, "size": 512, "axis": None}
    out = unwrap_batch([m], cfg)
    uv = out["results"][0]["uv"]
    assert len(uv) == 3
    for u, v in uv:
        assert -EPS <= u <= 1 + EPS
        assert -EPS <= v <= 1 + EPS


# --------------------------------------------------------------------------- #
# 组合 / 边界 4：旋转 90 度后面积不变（1e-6）
# --------------------------------------------------------------------------- #


def test_rotate_to_fit_preserves_area():
    uv = [(0.0, 0.0), (0.8, 0.1), (0.7, 0.6), (0.05, 0.5)]
    before = polygon_area(uv)
    out = rotate_to_fit(uv, (1.0, 1.0))
    assert out["rotation"] in (0, 90, 180, 270)
    after = polygon_area(out["uv"])
    assert after == pytest.approx(before, abs=1e-6)


# --------------------------------------------------------------------------- #
# 组合 / 边界 5：批量全成功时顺序不变
# --------------------------------------------------------------------------- #


def test_unwrap_batch_preserves_input_order():
    one = mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])  # 1 面 -> 3 个 UV
    two = quad_xy(1.0)  # 2 面 -> 6 个 UV
    three = mesh(  # 3 面 -> 9 个 UV
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]],
        [[0, 1, 2], [0, 2, 3], [1, 4, 2]],
    )
    cfg = {"angle_threshold": 60.0, "padding": 0.02, "size": 1024, "axis": None}
    out = unwrap_batch([one, two, three], cfg)
    results = out["results"]
    assert len(results) == 3, "批量结果必须与输入等长"
    assert [len(r["uv"]) for r in results] == [3, 6, 9], "结果顺序必须与输入一致"
