# PC → Jetson Nano Agent 交接说明

本文档用于 PC 端 Agent 与 Jetson Nano 端 Agent 之间的工程交接。
Jetson 端 Agent 应先阅读本文，再执行安装、路径检查和硬件测试。

## 1. 工程位置和目标

工程目录：

```text
puzzle-vision-simulator/
```

目标平台：

```text
Jetson Nano + USB 相机 + CH340/CH341 + STM32F103 下位机
```

目标是把 PC 上已经验证的视觉、拼接、坐标映射和比赛控制程序迁移到 Jetson Nano，保持下位机协议和动作逻辑不变。

## 2. PC 端已完成内容

- USB/UVC 摄像头采集，默认图像 1280×720。
- 摄像头画面默认旋转 180°。
- A4 ROI 框选和保存。
- 空桌面背景采集。
- 白色碎片分割、轮廓检测、凸包、顶点、中心、方向、安全抓取点。
- 普通白色碎片 2（1）使用 Git 节点 4.0 的核心拼接路径。
- 1（1）使用固定 4 个放置点，只做搬运，不调用拼接算法。
- 1（2）使用自备 4 块固定模板拼接，并支持旋转和间隙。
- 2（2）使用几何边匹配和牌面纹理接缝评分。
- 像素到龙门架 XY 脉冲标定。
- 舵机抓取角、放置角和 135°归位角已进入上位机任务帧。
- 串口 17 字节命令、B0/B1/B2/B3 状态解析。
- 自动连续执行只在收到 B1 后发送下一块。
- 串口读取异常会自动重连，但不会自动重发未知状态下的当前动作。

## 3. 当前关键配置

以下文件是 Jetson 运行必须存在的真实设备配置，已经解除 Git 忽略：

```text
configs/local/a4_roi.json
configs/local/calibration.json
configs/local/vision_detection.json
data/local/empty_work_area.png
```

临时标定和调试文件不是比赛运行必需品：

```text
configs/local/calibration_points_draft.json
configs/local/calibration_temporary.json
configs/local/piece_vision_debug.json
```

如果 Jetson 上的相机位置、分辨率、A4 位置或龙门架零点不同，必须在 Jetson 上重新采集 ROI、背景并重新标定，不能盲目沿用 PC 矩阵。

## 4. 路径适配状态

已新增：

```text
puzzle_device/paths.py
```

程序现在使用源码目录作为工程根目录，配置路径不再依赖当前终端所在目录。以下启动方式都应能找到配置：

```bash
cd ~/puzzle-vision-simulator
python3 -m apps.competition_gui --camera 0 --serial /dev/ttyUSB0
```

或者从其他目录调用绝对路径脚本。

## 5. Jetson 端环境检查

先执行：

```bash
python3 --version
python3 -c "import cv2, numpy, tkinter, serial; print(cv2.__version__)"
v4l2-ctl --list-devices
ls -l /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || true
```

建议安装 JetPack 系统包：

```bash
sudo apt update
sudo apt install python3-tk python3-opencv python3-numpy v4l-utils
python3 -m pip install -r requirements-jetson.txt
```

Jetson Nano 不建议直接使用 pip 安装 `opencv-python`，优先使用 JetPack 自带的 OpenCV。

## 6. 硬件设备约定

PC 端：

```text
相机：--camera 1
串口：COM30
```

Jetson 端通常是：

```text
相机：--camera 0，对应 /dev/video0
串口：/dev/ttyUSB0 或 /dev/ttyACM0
波特率：115200
```

如果串口权限不足：

```bash
sudo usermod -aG dialout "$USER"
```

执行后注销或重启，再重新测试。

## 7. 启动命令

比赛界面：

```bash
cd ~/puzzle-vision-simulator
python3 -m apps.competition_gui --camera 0 --serial /dev/ttyUSB0
```

也可以：

```bash
chmod +x 启动比赛.sh 启动标定.sh
./启动比赛.sh
```

标定界面：

```bash
python3 -m apps.manual_calibration_gui --camera 0 --serial /dev/ttyUSB0
```

识别调试界面：

```bash
python3 -m apps.piece_detection_gui --camera 0
```

## 8. Jetson 端测试顺序

不要一开始直接运行四块自动比赛，按以下顺序测试：

1. 确认相机能打开，画面为 1280×720，方向为 180°。
2. 确认 `configs/local/a4_roi.json` 能加载。
3. 确认背景图能加载；若光照变化，重新采集空桌面背景。
4. 确认视觉界面能框出碎片，并显示中心、顶点和安全抓取点。
5. 用标定界面测试几个像素点到脉冲的映射，确认不超 `X=0～2350、Y=0～1350`。
6. 执行串口回传测试：

   ```bash
   python3 -m tools.serial_return_test --serial /dev/ttyUSB0 --timeout 5
   ```

7. 先单块取放，确认 B0、B1 和舵机角度顺序正确。
8. 测试 1（1）四块固定搬运。
9. 测试 1（2）四块固定模板拼接。
10. 测试 2（1）先两块，再三块，最后四块。
11. 最后测试 2（2）扑克牌。

## 9. 协议和安全约束

Jetson 端 Agent 不得修改以下内容，除非得到 PC 端 Agent 明确确认：

- STM32 下位机代码。
- 17 字节命令帧格式。
- A2 双舵机角命令格式。
- B0/B1/B2/B3 状态含义。
- XY 回零和舵机 135°归位流程。
- 超出 `0～2350 / 0～1350` 的安全检查。

未知串口状态时禁止自动重发当前动作，防止机械重复抓取。

## 10. 遇到问题时回传给 PC Agent 的信息

请不要只发送“运行失败”，应回传：

```text
JetPack 版本：
Python 版本：
OpenCV 版本：
相机设备：
串口设备：
启动命令：
完整报错：
相机分辨率和 FPS：
是否收到 B0/B1/B2/B3：
```

如果是坐标问题，还要附上：

```text
碎片像素中心：
映射后的 X/Y 脉冲：
当前 ROI：
```

如果是视觉问题，保存并回传：

```text
output/assembly_vision_failed.json
output/assembly_vision_failed.png
output/assembly_vision_failed_overlay.png
```

## 11. 当前已知限制

- 当前标定矩阵只适用于建立它时的相机位置、分辨率和 A4 位置。
- Jetson 摄像头若实际输出不是 1280×720，必须重新标定。
- 2（2）目前主要依靠几何和纹理接缝，尚未加入完整 OCR、牌角数字对角校验和圆角语义判断。
- Jetson Nano 上需要实测识别耗时和比赛 120 秒限制。
- Tkinter 界面需要桌面显示环境，纯 SSH 无图形界面时不能直接打开 GUI。

## 12. 交接完成标准

Jetson 端 Agent 完成以下项目后，才算移植完成：

- 配置文件可以从源码目录正确加载；
- 相机稳定输出 1280×720；
- 串口能通过 B2 自检并收到 B0/B1；
- 单块取放成功；
- 1（1）四块连续执行成功；
- 1（2）拼接执行成功；
- 2（1）两块、三块、四块分别测试；
- 运行日志和失败诊断文件可以正常保存。

