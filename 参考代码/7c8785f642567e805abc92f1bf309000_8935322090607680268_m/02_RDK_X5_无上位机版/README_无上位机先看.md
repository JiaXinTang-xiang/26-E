# RDK X5 无上位机版

本目录与有上位机版使用同一份最新视觉算法，但默认只监听板内 `127.0.0.1`，不需要 Windows 电脑、浏览器、显示器或局域网。

## 手动启动

```bash
cd /home/sunrise/puzzle_vision
chmod +x *.sh
./启动无上位机模式.sh
```

切换模式：

```bash
START_MODE=unknown-white ./启动无上位机模式.sh
START_MODE=unknown-pattern ./启动无上位机模式.sh
```

## 安装开机自启

```bash
cd /home/sunrise/puzzle_vision
chmod +x *.sh
./安装无上位机开机自启.sh
```

指定普通白片模式和摄像头：

```bash
START_MODE=unknown-white \
CAMERA_SOURCE=usb:/dev/video2 \
./安装无上位机开机自启.sh
```

板内机械臂程序读取：

```text
http://127.0.0.1:8000/api/status
http://127.0.0.1:8000/api/export-motion.json
```

执行机械臂动作前必须实时确认 `motion_ready=true`，不能只根据磁盘上存在旧 JSON 就执行。

完整安全流程请阅读总包：

```text
05_说明书/RDK_X5无上位机独立运行指南.md
```
