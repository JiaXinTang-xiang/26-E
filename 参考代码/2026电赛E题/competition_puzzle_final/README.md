# 现场拼图比赛版（ROS1 / SwiftPro）

## 文件

- `a4_corner_calibrator.py`：鼠标点击 A4 有效黑纸四角并保存 YAML。
- `puzzle_piece_detector_fixed_corners_v4.py`：相机、透视、深度、手眼坐标基础模块。
- `onsite_puzzle_solver_v3.py`：现场分割、轮廓、纹理评分核心模块。
- `onsite_puzzle_solver_final.py`：比赛保护层；后台求解、4.5 秒超时、缓存、状态发布。
- `onsite_puzzle_executor_final.py`：SwiftPro 无第四轴执行节点。
- `competition_params.yaml`：识别和求解参数。
- `competition_executor_params.yaml`：机械臂参数。
- `competition_puzzle.launch`：启动模板，需替换包名。

## 与旧版相比

1. 求解放到后台线程，`rqt_image_view` 不会因递归搜索长期停住。
2. 默认每次求解最多 4.5 秒。
3. 同一组碎片未旋转时复用缓存，不重复求解。
4. 轮廓稳定 3 帧后才开始求解。
5. 限制矩形候选、放置候选、递归节点和分支数。
6. 不使用 `int.bit_count()`，兼容旧 Python 3。
7. 新增 `/puzzle/solver_status`：`STABILIZING`、`SOLVING`、`FOUND`、`TIMEOUT`、`NOT_FOUND`。

## 安装

假设功能包为 `your_pkg`：

```bash
mkdir -p ~/spark_noetic/src/your_pkg/scripts
mkdir -p ~/spark_noetic/src/your_pkg/config
mkdir -p ~/spark_noetic/src/your_pkg/launch

cp a4_corner_calibrator.py \
   puzzle_piece_detector_fixed_corners_v4.py \
   onsite_puzzle_solver_v3.py \
   onsite_puzzle_solver_final.py \
   onsite_puzzle_executor_final.py \
   ~/spark_noetic/src/your_pkg/scripts/

cp competition_params.yaml \
   competition_executor_params.yaml \
   ~/spark_noetic/src/your_pkg/config/

cp competition_puzzle.launch \
   ~/spark_noetic/src/your_pkg/launch/

chmod +x ~/spark_noetic/src/your_pkg/scripts/*.py
```

编辑 `competition_puzzle.launch`，把两处 `YOUR_PACKAGE` 改成 `your_pkg`。

```bash
cd ~/spark_noetic
catkin_make
source devel/setup.bash
```

## 赛前四角标定

```bash
rosrun your_pkg a4_corner_calibrator.py
```

依次点击：左上、右上、右下、左下，按 `S` 或 Enter 保存。默认文件：

```text
~/.ros/puzzle_a4_corners.yaml
```

## 启动

先启动相机和 SwiftPro 驱动：

```bash
roslaunch swiftpro pro_control.launch
```

再启动比赛节点：

```bash
roslaunch your_pkg competition_puzzle.launch
```

也可以只启动识别求解：

```bash
rosrun your_pkg onsite_puzzle_solver_final.py \
  _paper_corner_file:=$HOME/.ros/puzzle_a4_corners.yaml \
  _grasp_z:=-35 \
  _place_z:=-30
```

## 调试话题

```bash
rqt_image_view /puzzle/annotated_image
rqt_image_view /puzzle/warped_image
rqt_image_view /puzzle/warped_white_seed
rqt_image_view /puzzle/warped_piece_mask
rostopic echo /puzzle/solver_status
rostopic echo /puzzle/piece_coordinates
```

正确顺序通常是：

```text
STABILIZING -> SOLVING -> FOUND
```

`/puzzle/warped_piece_mask` 应只有 1~4 个完整白色多边形，背景应基本全黑。

## 执行

画面显示 `FOUND`，并且没有第四轴时每片剩余旋转角均不超过 18°，调用：

```bash
rosservice call /onsite_puzzle_executor/start
```

停止后续动作并关闭气泵：

```bash
rosservice call /onsite_puzzle_executor/stop
```

单独关闭气泵：

```bash
rosservice call /onsite_puzzle_executor/pump_off
```

完成后机械臂移动到：

```text
X=100 mm, Y=-100 mm, Z=35 mm
```

## 关键参数

正式现场碎片每条边不小于 20 mm，建议：

```yaml
detected_min_edge_mm: 14.0
```

此前题图中的固定测试碎片存在约 10 mm 短边，仅测试旧碎片时可临时改成：

```yaml
detected_min_edge_mm: 8.0
min_edge_overlap_mm: 6.0
```

比赛版默认求解限制：

```yaml
solver_timeout_sec: 4.5
max_rectangle_candidates: 12
packing_grid_mm: 2.0
packing_node_limit: 30000
```

求解经常 `TIMEOUT` 时，先检查轮廓是否正确；轮廓正确后可将：

```yaml
packing_grid_mm: 3.0
max_rectangle_candidates: 8
```

求解 `NOT_FOUND` 时，不建议先增加节点数，应先确认：

- 所有碎片均被识别；
- 每片多边形边数为 3~5；
- 碎片没有接触或粘连；
- 目标矩形总面积与碎片总面积接近；
- 每片确实至少有一条边属于目标矩形外边。

## 无第四轴限制

现场任意碎片通常需要较大旋转。当前机械臂第四轴无供电时，执行节点会拒绝超过 ±18° 的方案。此时需要人工预先摆正碎片；识别和求解仍会输出所需旋转角。要实现完全自动的随机姿态拼图，必须恢复第四轴供电或增加独立旋转机构。

## 安全

- 首次运行先将执行节点的 `dry_run` 改为 `true`。
- 确认 XY 对准后再使用 `pick_z=-35`。
- 不要同时运行其他发布 `position_write_topic` 或 `pump_topic` 的节点。
- 软件停止不能代替实体急停。
