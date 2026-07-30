# 电赛 E 题拼图图像生成、识别与仿真移动

本目录是一套可复现的 Python + OpenCV 仿真系统，覆盖：

- 题图规定的四片选手自备碎片；
- 测评现场纯白碎片；
- 测评现场白底扑克牌碎片；
- 图像分割、几何中心与安全抓取点计算；
- 不读取标准答案的自动拼图；
- A4 下半区数字移动过程和毫米坐标运动清单。

当前版本只进行图像仿真，不连接电机，也不定义 ESP32/STM32 串口协议。

## 环境

推荐 Python 3.12。安装依赖：

```powershell
python -m pip install -r requirements.txt
```

程序默认使用纵向 A4，尺寸为 `210 mm × 297 mm`，图像比例为
`5 px/mm`，即 `1050 × 1485 px`。图像坐标原点位于 A4 左上角，
X 轴向右，Y 轴向下，所有运动接口坐标均使用毫米。

画面中的暖白色矩形就是完整 A4，查看器外围使用深色以显示纸张边界。
纯白碎片比纸张底色略亮，并带有轻微投影和灰色切边，便于视频展示和
模拟白色碎片在白纸上的轮廓。

## 一键演示

生成并求解固定四片、现场纯白和现场扑克牌三组默认案例：

```powershell
python run_demo.py --seed 20260729 --field-count 4
```

生成结果位于 `images/`，识别与移动结果位于 `outputs/`。

## 可视化界面

双击：

```text
启动可视化界面.bat
```

或者在 PowerShell 中运行：

```powershell
python visual_app.py
```

界面支持：

- 选择选手自备、现场纯白或现场扑克牌模式；
- 设置现场碎片数量和随机种子；
- 一键生成、识别、拼图和数字移动；
- 打开已有的 A4 拼图输入图片；
- 缩放查看输入图、识别标注和最终拼图；
- 逐帧或自动播放碎片移动过程；
- 查看中心点、安全抓取点、目标位置和旋转角；
- 直接查看完整 `movement_plan.json`；
- 一键打开结果输出目录。

也可以启动后立即识别指定图片：

```powershell
python visual_app.py --input images/field_card_seed20260729_n4/input.png
```

## 分步命令

生成三种案例：

```powershell
python generate_puzzles.py --kind self --seed 20260729
python generate_puzzles.py --kind field-white --count 3 --seed 20260729
python generate_puzzles.py --kind field-card --count 4 --seed 20260729
```

现场模式的 `--count` 只能为 `1～4`。`self` 始终固定为四片。

只根据输入图片识别、求解并仿真移动：

```powershell
python solve_and_move.py `
  --input images/field_card_seed20260729_n4/input.png `
  --output outputs/field_card_seed20260729_n4
```

求解器只读取 `input.png`，不会根据目录名、随机种子或
`ground_truth.json` 推断答案。

## 每个生成案例

```text
images/<case>/
├─ input.png                 算法实际输入
├─ target_reference.png      测试用完整目标参考图
├─ ground_truth.json         仅供自动测试，不供求解器读取
└─ pieces/
   ├─ piece_01.png           透明背景独立碎片
   └─ ...
```

现场矩形宽度随机为 `90～120 mm`，高度随机为 `50～90 mm`。
每片不超过五条边、每条边不短于 `20 mm`，各片并集为完整矩形。
相同类型、种子和数量会生成相同图片。

## 每个求解结果

```text
outputs/<case>/
├─ detected.png              编号、轮廓、中心、抓取点和角度
├─ solved.png                移动到 A4 下半区后的结果
├─ movement_plan.json        机械平台可继续消费的毫米坐标
└─ movement_steps/
   ├─ step_00.png            检测结果
   ├─ step_01.png            第一片移动后
   └─ ...
```

`center_mm` 是多边形面积质心，用于描述碎片位姿；
`pick_point_mm` 是距离碎片边缘最远的内部点，更适合电磁头或吸盘抓取。
`rotation_deg` 是从输入姿态转到目标姿态所需的旋转量。

`movement_plan.json` 的核心字段如下：

```json
{
  "a4_size_mm": [210.0, 297.0],
  "target_rect": {
    "center_mm": [105.0, 222.75],
    "width_mm": 100.0,
    "height_mm": 60.0
  },
  "pieces": [
    {
      "id": 1,
      "source_center_mm": [60.0, 55.0],
      "pick_point_mm": [58.0, 57.0],
      "source_angle_deg": 32.0,
      "target_center_mm": [90.0, 220.0],
      "target_pick_point_mm": [89.0, 221.0],
      "target_angle_deg": 0.0,
      "translation_mm": [30.0, 165.0],
      "rotation_deg": -32.0,
      "sequence": 1
    }
  ]
}
```

后续接机械平台时，应先用相机标定把 A4 毫米坐标转换为机械坐标，再把
平移量和旋转量换算为电机步数。不要直接把图像像素当成电机坐标。

## 算法结构

- `puzzle_core/generation.py`：碎片、扑克牌纹理和随机摆放；
- `puzzle_core/vision.py`：前景分割、轮廓、多边形、中心和抓取点；
- `puzzle_core/solver.py`：固定模板匹配、共享边回溯和纹理接缝评分；
- `puzzle_core/simulation.py`：数字移动、步骤图片和运动 JSON；
- `puzzle_core/pipeline.py`：仅从输入图片执行完整求解流程。

扑克牌图案由程序绘制，不依赖外部图片。其求解先找几何可行解，再使用
接缝两侧的颜色连续性选择候选。纯白模式只使用几何关系。

## 测试

快速单元测试：

```powershell
python -m unittest discover -s tests -v
```

正式批量验证，两种现场模式、四种数量、每种 20 个随机种子，共
160 个案例：

```powershell
python batch_validate.py --seeds 20 --start-seed 2000
```

报告写入 `outputs/verification_report.json`。验证内容包括生成约束、
检测数量、模式、中心与角度误差、矩形尺寸、下半区位置、面积误差和耗时。

## 真实比赛前还要补的工作

仿真使用理想俯视图。真实装置至少还需要：

1. 相机内参标定和透视矫正；
2. A4 四角或 ArUco 标记定位；
3. 光照归一化和阴影处理；
4. 抓取头与相机的手眼标定；
5. 抓取失败检测、避障和二次定位；
6. ESP32/STM32 串口协议、限位与急停。

因此本项目适合先验证视觉和拼图算法，不应把仿真精度直接等同于实机精度。
