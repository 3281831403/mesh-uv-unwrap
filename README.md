# mesh-uv-unwrap

给一批模型自动展 UV 后，贴图不是拉伸就是串色：岛切分阈值拿弧度去和度数比，60 度的折边被当成 60 弧度于是从不切开；投影不选主轴，斜面侧壁上的纹理被明显拉伸；归一化对 u、v 各自缩放到 0..1，圆形贴图变成椭圆；打包不给岛之间留 padding，采样时相邻岛串色；旋转只试 0 度，长条岛横在图集里浪费大半空间；纹素密度不做归一，同一模型不同岛的密度差到 3 倍；UV 校验不查重叠，重叠区互相覆盖却查不出来；退化三角形不跳过，除法算出 NaN 污染整张 UV；批量展开所有网格共用一份 UV 缓冲，网格之间互相串数据；放不下的溢出岛被静默丢弃，最终 UV 少一块却没有任何提示。unwrap/atlas.py 是核心，tests/test_atlas.py 15个红。要修的是：按 docstring 写清的契约把 `unwrap/atlas.py` 里 8 个 `raise NotImplementedError` 的桩逐个实现，把上面 10 类缺陷全部修掉，让 `python -m pytest tests -q` 从 15 failed 变成 15 passed。

## 运行方式

```bash
cd mesh-uv-unwrap
python -m pytest tests -q
```

初始状态（未改动任何实现）必须恰好是：

```
15 failed in 0.12s
```

15 条用例全部红、且**没有** collection error / import error / syntax error——失败原因只能是
`NotImplementedError`。修完之后同样的命令应变成 `15 passed`。要求 Python 3，只装 pytest，
离线可跑，不使用随机数，重复运行结果一致。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

- **tests 不可修改。** `tests/test_atlas.py` 是验收标准，只能改 `unwrap/atlas.py`（以及
  `unwrap/__init__.py` 的导出）；不要为了让用例通过而改动、删除、跳过或重命名测试。
- 只用标准库（`math`、`heapq`）加 pytest，不引入 numpy 等第三方包。
- 网格数据结构固定为 `{"vertices": [[x, y, z], ...], "faces": [[i, j, k], ...]}`。
- UV 位于单位正方形 `0..1`；岛与岛、岛与图集边界之间必须留 `padding`；面积、长度、
  夹角比较容差统一 `1e-6`；角度参数以度为单位传入。
- 非法输入一律抛 `UnwrapError`，不允许静默丢弃数据（尤其是溢出的岛，必须登记在
  返回值的 `overflow` 里）。

## 文件结构

```
mesh-uv-unwrap/
├── README.md
├── conftest.py          # 把仓库根加入 sys.path，使 tests 可直接 import unwrap
├── .gitignore
├── unwrap/
│   ├── __init__.py
│   └── atlas.py         # 8 个桩 + UnwrapError，核心待实现文件
└── tests/
    └── test_atlas.py    # 15 条验收用例（不可修改）
```
