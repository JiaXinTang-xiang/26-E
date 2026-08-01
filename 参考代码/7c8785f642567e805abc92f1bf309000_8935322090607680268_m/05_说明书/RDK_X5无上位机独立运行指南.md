# RDK X5 无上位机独立运行指南

## 1. “无上位机运行”是什么意思

本方案不需要 Windows 电脑、浏览器或显示器长期连接。RDK X5 接好摄像头后，可独立完成：

1. 采集相机画面；
2. 识别 A4、分界线和 2～4 块碎片；
3. 求解下半区目标矩形；
4. 在方案通过校验时，输出吸取点、放置点和顺时针旋转角；
5. 通过板内 JSON 文件或本机 HTTP 接口把结果交给机械臂程序。

网页上位机只是可选的观察和调参工具。`vision_server.py` 即使没有任何浏览器访问，也会持续采图、识别和求解。

> 本交付包只负责视觉。机械臂坐标标定、真空泵、电机、限位和急停逻辑仍需由机械控制程序实现。

## 2. 推荐的独立运行结构

```text
USB/MIPI 摄像头
       │
       ▼
RDK X5 vision_server.py（开机自启、无浏览器）
       │
       ├─ /api/status：实时有效性和完整识别结果
       ├─ /api/export-motion.json：当前有效机械臂方案
       └─ exports/latest_motion_plan.json/csv：板内结果文件
                    │
                    ▼
       RDK X5 上的机械臂控制进程
```

推荐一直运行 `vision_server.py`，而不是反复调用单次识别。持续服务能够保留场景稳定判断和已验证拼法，碎片未移动时不会随意切换另一种拼接方案。

## 3. 首次检查

程序默认放在：

```text
/home/sunrise/puzzle_vision
```

在 RDK 终端执行：

```bash
cd /home/sunrise/puzzle_vision
python3 -c "import cv2, numpy; print('OpenCV', cv2.__version__)"
python3 main.py --config config.json self-test --output-dir self_test_output
v4l2-ctl --list-devices
```

然后检查摄像头节点：

```bash
for d in /dev/video*; do
  echo "===== $d ====="
  v4l2-ctl -d "$d" --list-formats-ext 2>/dev/null | head -n 25
done
```

应选能列出 `MJPG` 或 `YUYV` 及正常分辨率的节点，例如 `/dev/video0`。同一物理相机可能产生多个 `/dev/video*`，不要让两个进程同时占用同一采集通道。

## 4. 不安装开机服务时手动独立运行

基础固定题：

```bash
cd /home/sunrise/puzzle_vision
python3 vision_server.py \
  --host 127.0.0.1 \
  --port 8000 \
  --source usb:/dev/video0 \
  --mode fixed \
  --source-region upper \
  --use-color-hints
```

普通白片自主拼接：

```bash
python3 vision_server.py \
  --host 127.0.0.1 \
  --port 8000 \
  --source usb:/dev/video0 \
  --mode unknown-white \
  --source-region upper \
  --use-color-hints
```

扑克牌或花纹碎片：

```bash
python3 vision_server.py \
  --host 127.0.0.1 \
  --port 8000 \
  --source usb:/dev/video0 \
  --mode unknown-pattern \
  --source-region upper \
  --use-color-hints
```

`127.0.0.1` 表示只允许板内程序访问，不依赖外部网络。需要临时用另一台电脑查看时，改为 `--host 0.0.0.0`，浏览器访问 `http://RDK_IP:8000/`。

## 5. 设置开机自动运行

以下以基础模式、USB `/dev/video0` 为例。先创建 systemd 服务：

```bash
sudo tee /etc/systemd/system/a4-puzzle-vision.service >/dev/null <<'EOF'
[Unit]
Description=A4 Puzzle Vision on RDK X5
After=local-fs.target
ConditionPathExists=/home/sunrise/puzzle_vision/vision_server.py

[Service]
Type=simple
User=sunrise
WorkingDirectory=/home/sunrise/puzzle_vision
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3 /home/sunrise/puzzle_vision/vision_server.py --host 127.0.0.1 --port 8000 --source usb:/dev/video0 --mode fixed --source-region upper --use-color-hints
Restart=on-failure
RestartSec=2
KillSignal=SIGTERM
TimeoutStopSec=8

[Install]
WantedBy=multi-user.target
EOF
```

启用并立即启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now a4-puzzle-vision.service
systemctl status a4-puzzle-vision.service --no-pager
```

如果使用普通白片或扑克牌，把 `ExecStart` 中的模式分别改为：

```text
--mode unknown-white
--mode unknown-pattern
```

修改服务后执行：

```bash
sudo systemctl daemon-reload
sudo systemctl restart a4-puzzle-vision.service
```

常用管理命令：

```bash
sudo systemctl start a4-puzzle-vision.service
sudo systemctl stop a4-puzzle-vision.service
sudo systemctl restart a4-puzzle-vision.service
journalctl -u a4-puzzle-vision.service -f
```

## 6. 独立运行时怎样取得机械臂方案

### 6.1 实时状态

```bash
curl -s http://127.0.0.1:8000/api/status
```

机械臂只能在以下条件全部成立时读取动作：

- `motion_ready` 为 `true`；
- `motion_export_ready` 为 `true`；
- `error` 为空；
- `motion_export.commands` 数量等于当前碎片数；
- 每条命令的 `mirrored` 都为 `false`；
- 同一场景连续多次读取均为 READY，机械臂开始动作后立即进入“已消费、等待场景变化”状态。

### 6.2 获取标准机械臂 JSON

```bash
curl -f http://127.0.0.1:8000/api/export-motion.json \
  -o /tmp/current_motion_plan.json
```

也可取得 CSV：

```bash
curl -f http://127.0.0.1:8000/api/export-motion.csv \
  -o /tmp/current_motion_plan.csv
```

JSON 主要字段：

```json
{
  "schema": "a4-puzzle-motion-plan/v1",
  "motion_ready": true,
  "coordinate_frame": "a4_top_left_mm",
  "commands": [
    {
      "sequence": 1,
      "piece_id": "piece_1",
      "pick_a4_mm": [52.1, 43.8],
      "pick_camera_px": [603.2, 281.5],
      "place_a4_mm": [82.4, 216.7],
      "place_camera_px": [711.8, 892.4],
      "rotate_deg_clockwise": -31.5,
      "mirrored": false
    }
  ]
}
```

坐标规定：

- A4 毫米坐标原点是纸张左上角；
- X 向右，Y 向下；
- `rotate_deg_clockwise` 的正方向是顺时针；
- `pick_camera_px` 和 `place_camera_px` 是原始相机像素坐标；
- 机械臂应优先使用 A4 毫米坐标，并通过现场标定转换成机械坐标；
- `mirrored=true` 绝对不能执行，本题只允许平移和旋转。

### 6.3 板内结果文件

有效方案会自动写入：

```text
/home/sunrise/puzzle_vision/exports/latest_motion_plan.json
/home/sunrise/puzzle_vision/exports/latest_motion_plan.csv
```

**不能只判断文件是否存在。** 当后续画面无效、碎片被遮挡或相机断开时，磁盘上可能仍保留上一次的文件。执行前必须先查询 `/api/status`，确认实时 `motion_ready=true`，再读取 `/api/export-motion.json`。HTTP 导出接口在当前方案无效时会拒绝提供动作，安全性高于直接读取磁盘文件。

## 7. 推荐的机械臂触发状态机

无上位机运行时，机械臂控制程序应采用以下顺序：

1. **等待启动**：等待实体按键、GPIO 或上层控制器给出本轮开始信号；
2. **等待识别稳定**：连续读取 `/api/status`，至少连续 3 次满足 `motion_ready=true`；
3. **锁定方案**：读取一次 `/api/export-motion.json`，保存成本轮不可变副本；
4. **再次校验**：确认 `commands` 为 2～4 条、无镜像、目标点位于 A4 下半区；
5. **遮挡相机后不重算**：机械臂进入画面后只执行已经锁定的副本，不读取新方案；
6. **依次抓取**：按 `sequence` 执行吸取、旋转和放置；
7. **等待清场**：本轮方案标记为已消费，禁止相同结果重复执行；
8. **重新布片后再使能**：检测到画面由 READY 变为非 READY，或者收到新的实体启动信号后，才允许下一轮。

这样可以避免：

- 同一静止场景被机械臂重复执行；
- 机械臂遮挡画面时临时识别结果覆盖原方案；
- 读取到上一次遗留的 JSON；
- 多种合法拼法之间切换造成抓取目标变化。

## 8. 无上位机最小状态读取示例

下面程序只验证并打印方案，不控制电机。机械臂程序可复用这段标准库逻辑：

```python
import json
import time
from urllib.request import urlopen

BASE = "http://127.0.0.1:8000"
ready_count = 0

while ready_count < 3:
    with urlopen(BASE + "/api/status", timeout=2) as response:
        status = json.load(response)
    good = (
        status.get("motion_ready") is True
        and status.get("motion_export_ready") is True
        and not status.get("error")
    )
    ready_count = ready_count + 1 if good else 0
    time.sleep(0.25)

with urlopen(BASE + "/api/export-motion.json", timeout=2) as response:
    plan = json.load(response)

commands = plan.get("commands", [])
if not 2 <= len(commands) <= 4:
    raise RuntimeError("碎片动作数量不是 2～4")
if not plan.get("motion_ready"):
    raise RuntimeError("方案当前不可执行")
if any(command.get("mirrored") for command in commands):
    raise RuntimeError("检测到镜像动作，拒绝执行")

print(json.dumps(plan, ensure_ascii=False, indent=2))
# 从这里把 plan 固定为本轮副本，再交给机械臂控制模块。
```

## 9. 只在收到触发时做一次识别

如果现场不需要持续服务，也可在实体按键触发后运行一次：

```bash
cd /home/sunrise/puzzle_vision
python3 main.py --config config.json analyze \
  --source usb:/dev/video0 \
  --mode unknown-white \
  --source-region upper \
  --result result.json \
  --debug debug.jpg \
  --overlay overlay.jpg \
  --mask mask.png \
  --rectified rectified.jpg
```

返回码为 0 只表示程序正常完成，机械臂仍必须读取 `result.json` 并确认其中 `motion_ready=true`。单次方式每次都会重新打开相机，而且不能充分利用连续帧稳定确认和方案缓存，因此比赛正式运行更推荐第 5 节的常驻服务方式。

## 10. 无上位机故障恢复

服务状态：

```bash
systemctl is-active a4-puzzle-vision.service
curl -s http://127.0.0.1:8000/api/status
```

摄像头是否被占用：

```bash
fuser -v /dev/video0
```

最近日志：

```bash
journalctl -u a4-puzzle-vision.service -n 100 --no-pager
```

重启视觉服务：

```bash
sudo systemctl restart a4-puzzle-vision.service
```

建议机械控制程序设置以下故障原则：

- HTTP 连续超时或服务非 active：立即禁止抓取；
- `motion_ready=false`：保持等待，不执行旧文件；
- 相机断开：停止本轮并报警；
- 命令数超出 2～4、出现镜像或坐标越界：拒绝整轮方案；
- 机械臂急停和视觉服务应相互独立，视觉进程不能绕过硬件急停。

## 11. 比赛前离线独立运行检查表

- RDK 断开电脑和显示器后能正常开机；
- `a4-puzzle-vision.service` 显示 `active (running)`；
- 摄像头节点固定，重启后仍指向正确 `/dev/video*`；
- A4 四边、分界线和全部碎片均完整入镜；
- 三种模式中只启用与当前题目相符的一种；
- 连续三次状态均为 READY 后才允许机械臂取方案；
- 机械臂只执行本轮锁定副本，不在运动途中重新读方案；
- 清场或重新布片前不会重复执行旧方案；
- A4 毫米坐标到机械坐标已经完成至少 4 点标定；
- 断相机、遮挡、无分界线和识别失败时都能安全停止。
