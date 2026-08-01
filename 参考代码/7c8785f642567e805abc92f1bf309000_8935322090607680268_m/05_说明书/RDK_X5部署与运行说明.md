# RDK X5 部署与运行说明

## 1. 适用环境

- 地平线 RDK X5 / 旭日 X5，官方 Ubuntu 系统；
- Python 3；
- 系统 OpenCV 和 NumPy；
- UVC USB 摄像头；
- RDK 与操作电脑位于同一局域网。

运行前在 RDK 终端检查：

```bash
python3 --version
python3 -c "import cv2, numpy; print(cv2.__version__)"
v4l2-ctl --list-devices
ls -l /dev/video*
hostname -I
```

若没有 `v4l2-ctl`：

```bash
sudo apt update
sudo apt install -y v4l-utils
```

优先使用系统自带 OpenCV。不要直接用 `pip install opencv-python` 覆盖板卡镜像中的硬件适配版本。

## 2. Windows 一键部署

进入 `04_RDK_X5部署工具`。需要浏览器上位机时双击：

```text
一键部署RDK_X5_有上位机版.bat
```

脚本会：

1. 检查并安装 Windows 端 Paramiko；
2. 询问 RDK IP；
3. 询问初始识别模式；
4. 通过 SSH 上传 `01_RDK_X5_有上位机版`；
5. 在板端运行 Python 编译检查；
6. 停止旧视觉进程并释放 8000 端口；
7. 启动最新 `vision_server.py`；
8. 返回进程和日志。

默认参数：

```text
用户：sunrise
远程目录：/home/sunrise/puzzle_vision
端口：8000
```

密码由终端安全输入，不写入源码或压缩包。

命令行部署：

```bat
py -3 deploy_rdk.py ^
  --host 192.168.1.9 ^
  --project-dir "..\01_RDK_X5_有上位机版" ^
  --mode fixed
```

## 3. 手动复制部署

把 `01_RDK_X5_有上位机版` 整个目录复制到：

```text
/home/sunrise/puzzle_vision
```

然后执行：

```bash
cd /home/sunrise/puzzle_vision
python3 -m compileall -q puzzle_vision main.py vision_server.py
chmod +x 启动RDK_X5服务.sh 停止RDK_X5服务.sh
./启动RDK_X5服务.sh
```

浏览器打开：

```text
http://RDK_IP:8000/
```

## 4. 手动启动参数

基础模式：

```bash
python3 vision_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --source usb:/dev/video0 \
  --mode fixed \
  --source-region upper \
  --use-color-hints
```

普通白片：

```bash
START_MODE=unknown-white ./启动RDK_X5服务.sh
```

扑克牌：

```bash
START_MODE=unknown-pattern ./启动RDK_X5服务.sh
```

第二个摄像头通道：

```bash
CAMERA_SOURCE=usb:/dev/video2 ./启动RDK_X5服务.sh
```

## 5. 摄像头通道重复或打不开

一个物理 USB 摄像头可能暴露 `/dev/video0`、`/dev/video1` 等多个节点，其中部分节点只提供元数据。

检查格式：

```bash
for d in /dev/video*; do
  echo "===== $d ====="
  v4l2-ctl -d "$d" --list-formats-ext 2>/dev/null | head -n 30
done
```

选择能列出 `MJPG`、`YUYV` 且有目标分辨率的节点。若设备忙：

```bash
fuser -v /dev/video0
pkill -f '[p]ython3 vision_server.py'
fuser -k /dev/video0
```

再启动一次。不要同时让系统预览程序、另一个上位机和本程序占用同一节点。

## 6. 板端自检

不需要摄像头：

```bash
cd /home/sunrise/puzzle_vision
python3 main.py --config config.json self-test --output-dir self_test_output
```

期望：

```json
{"ok": true}
```

摄像头抓图：

```bash
python3 main.py capture --source usb:/dev/video0 --output capture.jpg
```

单图分析：

```bash
python3 main.py analyze \
  --source capture.jpg \
  --mode fixed \
  --source-region upper \
  --result result.json \
  --overlay overlay.jpg \
  --mask mask.png \
  --rectified rectified.jpg
```

## 7. 日志与状态

```bash
tail -f /home/sunrise/puzzle_vision/vision_server.log
pgrep -af 'python3 vision_server.py'
ss -lntp | grep 8000
curl http://127.0.0.1:8000/api/status
```

停止：

```bash
cd /home/sunrise/puzzle_vision
./停止RDK_X5服务.sh
```

## 8. 性能建议

- 基础题使用 `fixed`，不要用扑克牌模式代替；
- 普通无花纹片使用 `unknown-white`；
- 只有扑克牌或明显花纹时使用 `unknown-pattern`；
- `detection_max_dimension=960` 是速度与精度的平衡值；
- A4 四边精修在原分辨率运行，但仅在重新定位 A4 时执行；
- 场景不变时保留解算缓存，避免下方拼法跳变；
- CPU 温度过高会降频，比赛前检查散热。

## 9. 网络改变后的处理

查看 RDK 新 IP：

```bash
hostname -I
```

将浏览器地址、部署工具 IP 和 `upper_computer.py --host` 同步改为新地址。网页显示断线时先验证：

```bash
ping RDK_IP
curl http://RDK_IP:8000/api/status
```
