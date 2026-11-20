"""UV 展开核心。

约定
----
* 网格是普通 dict：``{"vertices": [[x, y, z], ...], "faces": [[i, j, k], ...]}``；
  ``faces`` 里的下标指向 ``vertices``，三角形顶点按逆时针（CCW）给出。
* UV 是 ``(u, v)`` 浮点二元组组成的列表，与传入的 ``faces`` 或顶点列表**按序一一对应**。
* UV 空间是**单位正方形 0..1**（左闭右闭）；越界即非法。
* 岛屿之间必须留 ``padding`` 宽的空白带，避免采样时相邻岛串色。
* 所有面积、长度、夹角的比较容差统一用 ``1e-6``。
* 角度参数（``angle_threshold`` 等）以**度**为单位传入；内部做几何计算时按需换算弧度。
* 纯标准库实现（``math``、``heapq``），不使用随机数，结果可复现。
* 非法输入（负 padding、退化三角形、空岛屿、无法打包等）抛 :class:`UnwrapError`，
  禁止静默丢弃数据。
"""

import heapq  # noqa: F401  # 约定允许的标准库依赖，保留以备打包策略扩展
import math

__all__ = [
    "UnwrapError",
    "split_islands",
    "project_island",
    "normalize_island",
    "pack_islands",
    "rotate_to_fit",
    "texel_density",
    "validate_uv",
    "unwrap_batch",
]

#: 全局比较容差。
EPS = 1e-6


class UnwrapError(Exception):
    """UV 展开过程中的非法输入、不可满足约束或需要整体回滚时抛出。

    约定：
    * ``padding < 0``、``size`` 非正、``angle_threshold`` 不在 (0, 180] 等参数错误直接抛；
    * 岛屿为空、面下标越界、退化三角形（面积 <= 1e-6）等几何错误直接抛；
    * 批量展开中任一网格失败时抛出，且调用方不得留下部分写入的结果（整批回滚）。
    """


# --------------------------------------------------------------------------- #
# 内部几何工具
# --------------------------------------------------------------------------- #


def _validate_mesh(mesh):
    """检查 mesh 字段是否齐全，返回 (vertices, faces)。"""
    if not isinstance(mesh, dict) or "vertices" not in mesh or "faces" not in mesh:
        raise UnwrapError("网格必须是含 'vertices' 与 'faces' 字段的 dict")
    vertices = mesh["vertices"]
    faces = mesh["faces"]
    if not isinstance(vertices, list) or not isinstance(faces, list):
        raise UnwrapError("vertices 与 faces 必须是列表")
    return vertices, faces


def _check_face(face, n_verts):
    """校验单个面：必须是三角形且下标全部合法。"""
    if len(face) != 3:
        raise UnwrapError("面必须是三角形，收到 %d 个顶点" % len(face))
    for idx in face:
        if not isinstance(idx, int) or isinstance(idx, bool) or not (0 <= idx < n_verts):
            raise UnwrapError("面下标越界或非整数: %r" % (idx,))


def _tri_normal(vertices, face):
    """返回三角形的（未归一化）叉积法线向量与 3D 面积。"""
    a = vertices[face[0]]
    b = vertices[face[1]]
    c = vertices[face[2]]
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx, ny, nz = (
        uy * vz - uz * vy,
        uz * vx - ux * vz,
        ux * vy - uy * vx,
    )
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    area = length / 2.0
    return (nx, ny, nz), length, area


def _tri_area_2d(a, b, c):
    return abs(
        (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
    ) / 2.0


def _bbox_2d(pts):
    us = [p[0] for p in pts]
    vs = [p[1] for p in pts]
    return min(us), min(vs), max(us), max(vs)


# --------------------------------------------------------------------------- #
# 1. 岛切分：角度阈值以度为单位，内部换算成弧度再比较
# --------------------------------------------------------------------------- #


def split_islands(mesh, angle_threshold):
    """按相邻三角形法线夹角把网格切成若干 UV 岛。

    参数
    ----
    mesh : dict
        ``{"vertices": [[x, y, z], ...], "faces": [[i, j, k], ...]}``。
    angle_threshold : float
        切分阈值，单位是**度**，取值范围 ``(0, 180]``。相邻两面法线夹角
        **大于**该阈值时在公共边处切开；等于阈值时仍视为同一岛。
        注意：这里的数值是度，几何计算里的法线夹角是弧度，必须换算后比较
        （例如 60 度对应 ``math.radians(60)``，而不是直接拿 60 去比弧度）。

    返回
    ----
    list[list[int]]
        每个元素是一个岛的面下标列表（指向 ``mesh["faces"]``），岛内按面下标升序；
        岛与岛之间按各自最小面下标升序排列。每个面必须恰好出现在一个岛里。
        空网格返回 ``[]``。

    异常
    ----
    UnwrapError
        ``angle_threshold`` 不在 ``(0, 180]``；网格缺字段；面下标越界或面不是三角形。
    """
    vertices, faces = _validate_mesh(mesh)
    if not isinstance(angle_threshold, (int, float)) or isinstance(angle_threshold, bool):
        raise UnwrapError("angle_threshold 必须是数值")
    if not (0.0 < float(angle_threshold) <= 180.0):
        raise UnwrapError("angle_threshold 必须在 (0, 180] 度之间，收到 %r" % angle_threshold)
    if not faces:
        return []

    n_verts = len(vertices)
    n_faces = len(faces)
    threshold_rad = math.radians(float(angle_threshold))

    # 每面的（未归一化）叉积法线；退化三角形（面积 <= EPS）直接抛，
    # 不允许 NaN 污染后续计算。保留原始叉积而非单位法线：精确 90 度的两个
    # 叉积点积在浮点上恒为精确的 0，精确共面恒为 |n1||n2|，没有单位化舍入。
    raw_normals = []
    for face in faces:
        _check_face(face, n_verts)
        normal, length, area = _tri_normal(vertices, face)
        if area <= EPS:
            raise UnwrapError("退化三角形（3D 面积 <= 1e-6），无法参与岛切分")
        raw_normals.append(normal)

    # 边 -> 共享该边的面下标；共享一条边（两个公共顶点）的面互为邻接。
    edge_faces = {}
    for fi, face in enumerate(faces):
        for edge in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = (edge[0], edge[1]) if edge[0] < edge[1] else (edge[1], edge[0])
            edge_faces.setdefault(key, []).append(fi)

    # 并查集：法线夹角 <= 阈值（严格大于才切）的邻面合并到同一岛。
    parent = list(range(n_faces))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 角度以度传入、必须换算成弧度再比较（60 度 == radians(60)，不是裸 60）。
    for shared in edge_faces.values():
        if len(shared) < 2:
            continue
        for pos in range(1, len(shared)):
            fi, fj = shared[0], shared[pos]
            ni, nj = raw_normals[fi], raw_normals[fj]
            dot = ni[0] * nj[0] + ni[1] * nj[1] + ni[2] * nj[2]
            len_i = math.sqrt(ni[0] ** 2 + ni[1] ** 2 + ni[2] ** 2)
            len_j = math.sqrt(nj[0] ** 2 + nj[1] ** 2 + nj[2] ** 2)
            cos_angle = dot / (len_i * len_j)
            # 仅钳制最后一位浮点误差；原始叉积保证精确正交/共面时比值精确，
            # 因此不需要额外角度容差——加了反而会吞掉真实的微小折角。
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle = math.acos(cos_angle)
            # 严格“大于”才切：等于阈值仍同岛。
            if angle <= threshold_rad:
                union(fi, fj)

    components = {}
    for fi in range(n_faces):
        components.setdefault(find(fi), []).append(fi)

    islands = [sorted(members) for members in components.values()]
    islands.sort(key=lambda members: members[0])
    return islands


# --------------------------------------------------------------------------- #
# 2. 主轴投影：丢弃主轴那一维，正交投影保持面积比例
# --------------------------------------------------------------------------- #

_AXIS_MAP = {
    "x": (1, 2, 1),
    "y": (0, 2, 1),
    "z": (0, 1, 1),
    "-x": (1, 2, -1),
    "-y": (0, 2, -1),
    "-z": (0, 1, -1),
}


def project_island(mesh, faces, axis):
    """把一个岛的顶点按主轴方向投影到 2D，保持面积比例。

    参数
    ----
    mesh : dict
        同 :func:`split_islands`。
    faces : list[int]
        要投影的岛的面下标列表。
    axis : str
        主轴，取值 ``"x"``、``"y"``、``"z"``（允许带正负号前缀，如 ``"-y"``）。
        投影时丢弃 ``axis`` 这一维，取另外两维为 ``(u, v)``；负号表示该轴取反。
        调用方（或本函数）应依据岛的法线选择**模最大的分量**所在轴，而不是
        固定轴；固定轴会让斜面/侧壁上的纹理被拉伸。

    返回
    ----
    dict
        ``{"uv": [(u, v), ...], "axis": axis, "area3d": float, "area2d": float}``。
        ``uv`` 与岛引用到的唯一顶点按顶点下标升序一一对应（共享顶点只投影一次；
        需要按面角点展开的调用方自行依据 faces 索引取点）。
        投影是正交投影，因此 ``area2d / area3d`` 即该主轴方向的面积比例，
        应当在 ``1e-6`` 内等于 ``cos(法线与主轴夹角)`` 而不是任意值。

    异常
    ----
    UnwrapError
        ``faces`` 为空、``axis`` 非法、面下标越界、或岛退化为零面积。
    """
    vertices, all_faces = _validate_mesh(mesh)
    if not isinstance(faces, list) or not faces:
        raise UnwrapError("投影岛屿的 faces 不能为空")
    if axis not in _AXIS_MAP:
        raise UnwrapError("非法主轴 %r，只允许 +/- x/y/z" % (axis,))
    dim_u, dim_v, sign = _AXIS_MAP[axis]
    n_verts = len(vertices)

    referenced = set()
    for fi in faces:
        if not isinstance(fi, int) or isinstance(fi, bool) or not (0 <= fi < len(all_faces)):
            raise UnwrapError("岛面下标越界: %r" % (fi,))
        face = all_faces[fi]
        _check_face(face, n_verts)
        referenced.update(face)

    # 共享顶点只投影一次：uv 按被引用顶点的下标升序排列，每个唯一顶点一个点。
    ordered_idx = sorted(referenced)
    pos = {idx: k for k, idx in enumerate(ordered_idx)}
    uv = [
        (sign * vertices[idx][dim_u], vertices[idx][dim_v])
        for idx in ordered_idx
    ]

    area3d = 0.0
    area2d = 0.0
    for fi in faces:
        face = all_faces[fi]
        _, _, tri_area = _tri_normal(vertices, face)
        area3d += tri_area
        a, b, c = (pos[face[0]], pos[face[1]], pos[face[2]])
        area2d += _tri_area_2d(uv[a], uv[b], uv[c])

    if area3d <= EPS:
        raise UnwrapError("岛屿 3D 总面积 <= 1e-6，退化为零面积，无法投影")

    return {"uv": uv, "axis": axis, "area3d": area3d, "area2d": area2d}


# --------------------------------------------------------------------------- #
# 3. 单岛归一化：等比缩放，保持长宽比（圆不能变椭圆）
# --------------------------------------------------------------------------- #


def normalize_island(uv):
    """把单个岛的 UV 归一化到 ``0..1``，且**保持长宽比**。

    参数
    ----
    uv : list[tuple[float, float]]
        岛的 UV 坐标，至少 3 个点。

    返回
    ----
    list[tuple[float, float]]
        与输入等长、同序的新列表。包围盒平移并等比缩放到单位正方形内：
        较长的一边贴到 0..1，较短的一边贴 0 留白，**不能分别**对
        u、v 独立缩放到 0..1，否则圆形会变成椭圆、长宽比失真。

    异常
    ----
    UnwrapError
        点数少于 3、或包围盒在某一维上宽度 <= 1e-6（退化，不可归一化）。
    """
    if not isinstance(uv, list) or len(uv) < 3:
        raise UnwrapError("归一化至少需要 3 个 UV 点")
    u0, v0, u1, v1 = _bbox_2d(uv)
    width = u1 - u0
    height = v1 - v0
    if width <= EPS or height <= EPS:
        raise UnwrapError("包围盒某一维宽度 <= 1e-6，岛屿退化，不可归一化")

    # 两维共用同一个缩放因子：保持长宽比。
    scale = 1.0 / max(width, height)
    return [((u - u0) * scale, (v - v0) * scale) for (u, v) in uv]


# --------------------------------------------------------------------------- #
# 4. 图集打包：外扩盒互斥 + padding 边界 + 溢出显式登记
# --------------------------------------------------------------------------- #


def pack_islands(islands, padding, size):
    """把若干已归一化的岛矩形打包进 ``size x size`` 的图集，返回整图 UV。

    参数
    ----
    islands : list[list[tuple[float, float]]]
        每个元素是一个岛的 UV 点列（各自已在 0..1 内）。
    padding : float
        岛与岛之间、岛与图集边界之间必须保留的最小空白宽度，``>= 0``。
        打包时每个岛占用的矩形是 ``[u0 - padding, v0 - padding, u1 + padding, v1 + padding]``
        这一外扩盒，相邻外扩盒不得相交，否则采样会串色。
    size : int | float
        图集边长（像素或单位长度），必须 > 0；返回的 UV 仍归一化到 0..1。

    返回
    ----
    dict
        ``{"uv": [...], "overflow": [...]}``：

        * ``uv``：所有岛的 UV 按输入岛顺序拼接，平移到各自打包位置后的结果；
        * ``overflow``：放不进图集的岛的**下标列表**（指向 ``islands``），按升序。
          溢出的岛必须在 ``uv`` 里被跳过、同时**显式登记**在 ``overflow``，
          不能静默丢弃——否则最终 UV 会少一块，且调用方无从知道。

    异常
    ----
    UnwrapError
        ``padding < 0``、``size <= 0``、``islands`` 为空、单岛包围盒超过图集可用区域。
    """
    if not isinstance(islands, list) or not islands:
        raise UnwrapError("待打包岛屿列表不能为空")
    if not isinstance(padding, (int, float)) or isinstance(padding, bool) or padding < 0:
        raise UnwrapError("padding 必须 >= 0，收到 %r" % (padding,))
    if not isinstance(size, (int, float)) or isinstance(size, bool) or size <= 0:
        raise UnwrapError("size 必须 > 0，收到 %r" % (size,))

    padding = float(padding)
    # 可用内容区 [padding, 1-padding]；若内容区不存在则任何岛都无法打包。
    if 1.0 - 2.0 * padding <= EPS:
        raise UnwrapError("padding 过大，图集没有可放置岛屿的内容区")

    rects = []  # (width, height, min_u, min_v)
    for pts in islands:
        u0, v0, u1, v1 = _bbox_2d(pts)
        w, h = u1 - u0, v1 - v0
        if w > 1.0 + EPS or h > 1.0 + EPS:
            # 单岛包围盒本身已超过单位图集，任何缩放外的手段都无法满足。
            raise UnwrapError("单岛包围盒超过图集可用区域")
        rects.append((w, h, u0, v0))

    def shelf_pack(scale):
        """单行 shelf 打包：外扩盒沿 u 轴顺序排列，水平放不下即溢出。

        不向下开新行：两个宽度之和超过图集的岛即使垂直方向有空隙也算放不下，
        由调用方决定是登记溢出还是整体缩小后重排。
        """
        placements = [None] * len(rects)
        overflow = []
        x = 0.0
        for i, (w, h, _, _) in enumerate(rects):
            rw = w * scale + 2.0 * padding
            rh = h * scale + 2.0 * padding
            if rh > 1.0 + EPS or x + rw > 1.0 + EPS:
                # 放不下：显式登记，绝不静默丢弃；也不挪动 x（本岛不占位）。
                overflow.append(i)
                continue
            placements[i] = (x, 0.0)
            x += rw
        return placements, overflow

    # 原生尺度先打一次。
    placements, overflow = shelf_pack(1.0)
    native_oversized = any(
        w + 2.0 * padding > 1.0 + EPS or h + 2.0 * padding > 1.0 + EPS
        for (w, h, _, _) in rects
    )
    scale = 1.0
    if overflow and native_oversized:
        # 存在“单独一个外扩盒就超过图集”的岛：整体等比缩小后重排，
        # 直到全部放下（纯确定性折半，不使用随机数）。
        for _ in range(100):
            scale *= 0.5
            placements, overflow = shelf_pack(scale)
            if not overflow:
                break
        if overflow:
            raise UnwrapError("岛屿无法打包进图集")
    # 若每个岛外扩盒单独都放得下、只是图集被占满：保持原生尺度，登记溢出。

    uv_out = []
    for i, (w, h, min_u, min_v) in enumerate(rects):
        if placements[i] is None:
            continue  # 溢出岛跳过其 UV，但已登记在 overflow 中
        px, py = placements[i]
        for (u, v) in islands[i]:
            uv_out.append(
                (
                    px + padding + (u - min_u) * scale,
                    py + padding + (v - min_v) * scale,
                )
            )

    return {"uv": uv_out, "overflow": overflow}


# --------------------------------------------------------------------------- #
# 5. 90 度整数倍旋转：四个朝向里选能放进 box 且包围盒面积最小的
# --------------------------------------------------------------------------- #

_ROTATIONS = (0, 90, 180, 270)


def _rotate_point(u, v, rotation):
    if rotation == 0:
        return u, v
    if rotation == 90:
        return -v, u
    if rotation == 180:
        return -u, -v
    return v, -u  # 270


def rotate_to_fit(island, box):
    """按 90 度整数倍旋转岛，选包围盒最小的朝向。

    参数
    ----
    island : list[tuple[float, float]]
        岛的 UV 点列。
    box : tuple[float, float]
        可用区域 ``(width, height)``，旋转后的包围盒必须落进去。

    返回
    ----
    dict
        ``{"uv": [(u, v), ...], "rotation": int, "bbox": (w, h)}``：
        ``rotation`` 是实际采用的旋转角，取值 ``0``、``90``、``180``、``270``；
        ``bbox`` 是旋转后（未平移）的包围盒宽高。
        **只试 0 度是不够的**：长条岛旋转 90 度后往往能省一半空间，
        所以必须在 0/90/180/270 里挑包围盒面积最小、且能放进 ``box`` 的那个；
        面积相同时优先能放下的、再按角度升序取第一个。
        旋转是正交变换，旋转前后岛的面积必须在 ``1e-6`` 内不变。

    异常
    ----
    UnwrapError
        点列少于 3、``box`` 非正、或四个朝向都放不下。
    """
    if not isinstance(island, list) or len(island) < 3:
        raise UnwrapError("旋转岛屿至少需要 3 个 UV 点")
    if (
        not isinstance(box, (tuple, list))
        or len(box) != 2
        or box[0] <= 0
        or box[1] <= 0
    ):
        raise UnwrapError("box 必须是正数的 (width, height)")
    box_w, box_h = box

    candidates = []
    for rotation in _ROTATIONS:
        rotated = [_rotate_point(u, v, rotation) for (u, v) in island]
        u0, v0, u1, v1 = _bbox_2d(rotated)
        w, h = u1 - u0, v1 - v0
        fits = w <= box_w + EPS and h <= box_h + EPS
        if fits:
            candidates.append((w * h, rotation, rotated, (w, h)))

    if not candidates:
        raise UnwrapError("四个旋转朝向都无法放进可用区域 %r" % (box,))

    # 面积最小；面积相同按角度升序取第一个（候选本身按 0/90/180/270 进入）。
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, rotation, rotated, bbox = candidates[0]
    return {"uv": rotated, "rotation": rotation, "bbox": bbox}


# --------------------------------------------------------------------------- #
# 6. 纹素密度：总 UV 面积 / 总 3D 面积，退化三角形贡献 0 而不是 NaN
# --------------------------------------------------------------------------- #


def texel_density(uv, mesh):
    """计算每单位 3D 面积的纹素密度，用于跨岛一致性校验。

    参数
    ----
    uv : list[tuple[float, float]]
        与 ``mesh`` 顶点按序对应的 UV。
    mesh : dict
        同 :func:`split_islands`。

    返回
    ----
    float
        ``sum(uv 三角形面积) / sum(3D 三角形面积)``（按 0..1 归一化的 UV 面积，
        再乘以图集分辨率才是像素意义下的密度）。分母小于 ``1e-6`` 时抛
        :class:`UnwrapError`。返回值必须是**归一化后的**密度：同一网格不同岛
        之间该值应彼此接近（差异远小于 3 倍），否则说明某个岛被单独缩放过。

    异常
    ----
    UnwrapError
        UV 与顶点数量不一致、或 3D 总面积 <= 1e-6。
    """
    vertices, faces = _validate_mesh(mesh)
    if len(uv) != len(vertices):
        raise UnwrapError(
            "UV 点数 %d 与顶点数 %d 不一致" % (len(uv), len(vertices))
        )
    for face in faces:
        _check_face(face, len(vertices))

    area_3d = 0.0
    area_uv = 0.0
    for face in faces:
        a, b, c = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        _, _, tri3 = _tri_normal([a, b, c], (0, 1, 2))
        # 退化三角形面积为 0：累加 0 即可，不做任何除法，杜绝 NaN/inf。
        area_3d += tri3
        area_uv += _tri_area_2d(uv[face[0]], uv[face[1]], uv[face[2]])

    if area_3d <= EPS:
        raise UnwrapError("3D 总面积 <= 1e-6，纹素密度分母非法")
    return area_uv / area_3d


# --------------------------------------------------------------------------- #
# 7. UV 校验：越界 / 重叠 / 退化，全覆盖且不产生 NaN
# --------------------------------------------------------------------------- #


def _polygon_area(pts):
    if len(pts) < 3:
        return 0.0
    acc = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        acc += x1 * y2 - x2 * y1
    return abs(acc) / 2.0


def _clip_polygon(poly, edge_start, edge_end):
    """Sutherland-Hodgman：用有向边 s->e 的左侧半平面裁剪多边形。"""
    if not poly:
        return []
    ex, ey = edge_end[0] - edge_start[0], edge_end[1] - edge_start[1]

    def side(pt):
        return ex * (pt[1] - edge_start[1]) - ey * (pt[0] - edge_start[0])

    out = []
    prev = poly[-1]
    prev_inside = side(prev) >= -EPS
    for cur in poly:
        cur_inside = side(cur) >= -EPS
        if cur_inside != prev_inside:
            d1, d2 = side(prev), side(cur)
            denom = d1 - d2
            if abs(denom) <= EPS:
                out.append(cur)
            else:
                t = d1 / denom
                out.append(
                    (
                        prev[0] + (cur[0] - prev[0]) * t,
                        prev[1] + (cur[1] - prev[1]) * t,
                    )
                )
        if cur_inside:
            out.append(cur)
        prev, prev_inside = cur, cur_inside
    return out


def _tri_intersection_area(tri_a, tri_b):
    """把 tri_a 依次用 tri_b 的三条边裁剪，剩余多边形面积即交集面积。"""
    poly = list(tri_a)
    tri_b = list(tri_b)
    for k in range(3):
        poly = _clip_polygon(poly, tri_b[k], tri_b[(k + 1) % 3])
        if not poly:
            return 0.0
    return _polygon_area(poly)


def validate_uv(uv):
    """检查 UV 的问题，返回人类可读的问题字符串列表（无问题返回 ``[]``）。

    参数
    ----
    uv : list[tuple[float, float]]
        待检查的 UV 点列。约定按三角形每 3 个点一组给出。

    返回
    ----
    list[str]
        每条是一个问题描述，按发现顺序排列。至少必须覆盖：

        * 越界：任一分量 ``< -1e-6`` 或 ``> 1 + 1e-6``；
        * 重叠：两个非相邻三角形在 UV 空间内有面积交集（交集面积 > 1e-6），
          重叠区会互相覆盖；
        * 退化三角形：三点共线或面积 <= 1e-6（这类三角形必须被跳过/报出，
          不能让除法产生 NaN 或 inf）。

        同一类问题在列表里可以出现多条；``[]`` 表示完全合法。

    异常
    ----
    UnwrapError
        ``uv`` 长度为 0 或不是 3 的倍数。
    """
    if not isinstance(uv, list) or len(uv) == 0:
        raise UnwrapError("UV 点列为空")
    if len(uv) % 3 != 0:
        raise UnwrapError("UV 点数 %d 不是 3 的倍数" % len(uv))

    issues = []
    triangles = [uv[i:i + 3] for i in range(0, len(uv), 3)]
    degenerate = [False] * len(triangles)

    for ti, tri in enumerate(triangles):
        for pi, (u, v) in enumerate(tri):
            if (
                isinstance(u, float) and math.isnan(u)
            ) or (
                isinstance(v, float) and math.isnan(v)
            ) or u < -EPS or u > 1.0 + EPS or v < -EPS or v > 1.0 + EPS:
                issues.append(
                    "三角形 %d 的顶点 %d 越界: (%r, %r)" % (ti, pi, u, v)
                )
        area = _tri_area_2d(tri[0], tri[1], tri[2])
        if area <= EPS:
            degenerate[ti] = True
            issues.append("三角形 %d 退化：面积 <= 1e-6（共线或重合）" % ti)

    # 重叠：只检查非相邻（不共享顶点）且都非退化的三角形对。
    for i in range(len(triangles)):
        if degenerate[i]:
            continue
        for j in range(i + 1, len(triangles)):
            if degenerate[j]:
                continue
            shared = any(
                abs(p[0] - q[0]) <= EPS and abs(p[1] - q[1]) <= EPS
                for p in triangles[i]
                for q in triangles[j]
            )
            if shared:
                continue
            overlap = _tri_intersection_area(triangles[i], triangles[j])
            if overlap > EPS:
                issues.append(
                    "三角形 %d 与三角形 %d 在 UV 空间重叠，交集面积 %.6g"
                    % (i, j, overlap)
                )

    return issues


# --------------------------------------------------------------------------- #
# 8. 批量展开：每网格独立缓冲，失败整批回滚，溢出显式汇总
# --------------------------------------------------------------------------- #

_DEFAULT_CONFIG = {
    "angle_threshold": 60.0,
    "padding": 0.01,
    "size": 1024,
    "axis": None,
}


def _dominant_axis(mesh, faces):
    """按面积加权平均法线选模最大分量所在轴（含正负号）。"""
    vertices, _ = _validate_mesh(mesh)
    nx = ny = nz = 0.0
    for fi in faces:
        normal, _, _ = _tri_normal(vertices, mesh["faces"][fi])
        nx += normal[0]
        ny += normal[1]
        nz += normal[2]
    components = (("x", nx), ("y", ny), ("z", nz))
    name, value = max(components, key=lambda item: abs(item[1]))
    return ("-" + name) if value < 0 else name


def _unwrap_one(mesh, config):
    """展开单个网格；返回该网格独立持有的全新 UV 缓冲与元数据。"""
    angle_threshold = config["angle_threshold"]
    padding = config["padding"]
    size = config["size"]
    forced_axis = config["axis"]

    islands_faces = split_islands(mesh, angle_threshold)
    if not islands_faces:
        return {"uv": [], "islands": [], "axes": [], "overflow": []}

    normalized_islands = []
    axes = []
    # 记录每岛投影点 -> 岛内面角点的展开顺序，便于打包后还原成按面对应的 UV。
    expansions = []
    for faces in islands_faces:
        axis = forced_axis if forced_axis is not None else _dominant_axis(mesh, faces)
        projected = project_island(mesh, faces, axis)
        axes.append(axis)
        referenced = sorted({idx for fi in faces for idx in mesh["faces"][fi]})
        pos = {idx: k for k, idx in enumerate(referenced)}
        order = [pos[idx] for fi in faces for idx in mesh["faces"][fi]]
        expansions.append(order)
        normalized_islands.append(normalize_island(projected["uv"]))

    packed = pack_islands(normalized_islands, padding, size)
    placed = packed["uv"]

    # 把按唯一顶点排列的打包结果还原成按面角点排列（每面 3 个点，与 faces 同序）。
    uv_out = []
    cursor = 0
    overflow_set = set(packed["overflow"])
    for island_idx, island_uv in enumerate(normalized_islands):
        if island_idx in overflow_set:
            # 溢出岛跳过；其下标已在 overflow 中显式登记，不静默丢失。
            continue
        for local_pos in expansions[island_idx]:
            uv_out.append(placed[cursor + local_pos])
        cursor += len(island_uv)

    # 每个网格各自构造全新列表，绝不与其他网格共享缓冲。
    return {
        "uv": uv_out,
        "islands": islands_faces,
        "axes": axes,
        "overflow": list(packed["overflow"]),
    }


def unwrap_batch(meshes, config):
    """批量展开多个网格，任何一个失败都必须整批回滚。

    参数
    ----
    meshes : list[dict]
        待展开的网格列表，元素同 :func:`split_islands`。
    config : dict
        配置项，至少可含 ``{"angle_threshold": 60.0, "padding": 0.01,
        "size": 1024, "axis": None}``；缺省项按上述默认值处理。

    返回
    ----
    dict
        ``{"results": [...], "overflow": [...]}``：
        ``results`` 与 ``meshes`` **等长同序**，每项是该网格自己的展开结果
        （含 ``uv``、``islands`` 等字段）。每个网格必须持有**独立**的 UV 缓冲，
        不允许共用同一份缓存，否则网格之间 UV 会互相串。
        ``overflow`` 汇总所有网格溢出的岛 ``(mesh_index, island_index)``。

        原子性：只要有一个网格非法，就抛 :class:`UnwrapError` 并且**不返回任何
        部分结果**；调用方已写入的状态也要能回滚，不能留下"前 3 个成功、第 4 个
        失败"的半成品。

    异常
    ----
    UnwrapError
        任一网格非法、``config`` 参数非法、或整批无法打包。
    """
    if not isinstance(meshes, list):
        raise UnwrapError("meshes 必须是网格 dict 的列表")
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise UnwrapError("config 必须是 dict")

    merged = dict(_DEFAULT_CONFIG)
    merged.update(config)

    # 配置校验前置：配置非法时一个网格都不处理。
    at = merged["angle_threshold"]
    if not isinstance(at, (int, float)) or isinstance(at, bool) or not (0.0 < at <= 180.0):
        raise UnwrapError("config.angle_threshold 必须在 (0, 180] 度之间")
    padding = merged["padding"]
    if not isinstance(padding, (int, float)) or isinstance(padding, bool) or padding < 0:
        raise UnwrapError("config.padding 必须 >= 0")
    size = merged["size"]
    if not isinstance(size, (int, float)) or isinstance(size, bool) or size <= 0:
        raise UnwrapError("config.size 必须 > 0")
    if merged["axis"] is not None and merged["axis"] not in _AXIS_MAP:
        raise UnwrapError("config.axis 非法: %r" % (merged["axis"],))

    # 先在局部把所有结果构造完：任一网格抛错则本函数没有返回值、也没有任何
    # 外部状态被修改，天然满足整批原子回滚。
    results = []
    overflow_all = []
    for mesh_index, mesh in enumerate(meshes):
        result = _unwrap_one(mesh, merged)
        for island_index in result["overflow"]:
            overflow_all.append((mesh_index, island_index))
        results.append(result)

    return {"results": results, "overflow": overflow_all}
