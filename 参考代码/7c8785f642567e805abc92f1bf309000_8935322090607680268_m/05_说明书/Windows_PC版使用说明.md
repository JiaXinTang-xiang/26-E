# Windows PC 版使用说明

## 1. 环境要求

- Windows 10 或 Windows 11，64 位；
- Python 3.10、3.11 或 3.12，64 位；
- USB/UVC 摄像头；
- 推荐 4 核以上 CPU、8 GB 内存；
- 浏览器推荐 Edge 或 Chrome。

安装 Python 时勾选 `Add Python to PATH`。不要安装 Microsoft Store 的受限 Python。

## 2. 首次安装

进入 `03_Windows_PC版`，双击：

```text
安装依赖_首次运行.bat
```

脚本安装：

- NumPy；
- OpenCV Python。

安装完成后双击：

```text
运行完整自检.bat
```

控制台应显示 `"ok": true`，且所有 `checks` 为 `true`。

## 3. 日常启动

双击：

```text
启动电脑版上位机.bat
```

默认参数：

- 摄像头：`usb:0`；
- 初始模式：`fixed`；
- 源区域：上半区；
- 地址：`http://127.0.0.1:8000/`。

启动脚本会等待服务真正就绪后再打开浏览器，避免网页先打开却显示断线。

关闭时回到黑色命令窗口按 `Ctrl+C`。不要直接反复双击启动，否则 8000 端口会被旧进程占用。

## 4. 选择其他摄像头或模式

在当前目录地址栏输入 `cmd` 后执行：

```bat
启动电脑版上位机.bat usb:1 fixed
启动电脑版上位机.bat usb:2 unknown-white
启动电脑版上位机.bat usb:0 unknown-pattern
```

也可以直接运行：

```bat
py -3 upper_computer_pc.py --source usb:1 --mode unknown-white
```

离线图片：

```bat
py -3 upper_computer_pc.py --source "D:\test\scene.jpg" --mode unknown-pattern
```

网页中的“上传图片识别”更适合临时测试，因为上传图会强制清除实时摄像头的 A4 坐标缓存。

## 5. 网页操作顺序

1. 选择识别方式。
2. 确认碎片在上半区、下半区或自动区域。
3. 按现场颜色设置 A4 底纸和碎片颜色。
4. 颜色明确时勾选“优先按指定颜色寻找”。
5. 先观察绿色 A4 框和橙色分界线。
6. 确认每块碎片颜色框、重心和吸取点。
7. 检查下方矩形内每块颜色是否与源碎片一一对应。
8. 状态为 `MOTION READY` 后导出机械臂 JSON 或 CSV。

## 6. 摄像头画面不完整

- 点击画面右上角“完整画面”；
- 浏览器缩放设为 80%～100%；
- Windows 显示缩放推荐 100% 或 125%；
- 摄像头尽量输出 1920×1080；
- 不要让浏览器开发者工具占据右侧空间。

## 7. Windows 防火墙

本机访问 `127.0.0.1` 通常无需开放端口。若局域网其他电脑需要访问，在 Windows 防火墙允许 Python 的“专用网络”，并使用：

```bat
py -3 vision_server.py --host 0.0.0.0 --port 8000 --source usb:0
```

然后访问 `http://电脑IP:8000/`。只在可信局域网中开放。

## 8. 文件输出

网页导出的机械臂文件包含：

- A4 左上角为原点的毫米坐标；
- 每块碎片安全吸取点；
- 目标放置点；
- 顺时针旋转角；
- `mirrored=false`；
- 目标多边形和无重叠校验。

程序运行生成的临时结果放在 `exports` 或指定输出目录，不属于源代码。
