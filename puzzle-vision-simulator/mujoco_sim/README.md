# MuJoCo 全流程拼图仿真

这是与原视觉程序隔离的新增目录，不修改原有 `puzzle_sim.py` 和 `puzzle_gui.py`
的使用方式。

仿真包含：

- 与控制站相同的六轴 PIPER-L 机械臂、数值逆解和五次时间轨迹规划；
- 末端电磁铁，以可动态启停的刚性磁吸约束模拟吸附和释放；
- 带实体外壳的俯视相机、主场景视角和相机视野小窗；
- A4 纸、上下分界线和 10 cm × 6 cm 目标框；
- 2～4 块随机多边形碎片；
- 相机图像分割、轮廓识别、拼接规划；
- 逐块接近、下降、吸附、抬升、搬运、释放的完整状态机。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r mujoco_sim/requirements.txt
```

## 运行

在仓库根目录执行：

```bash
./mujoco_sim/run_ui.sh
```

启动后会同时打开原视觉算法主窗口和独立的 MuJoCo 三维窗口。主窗口可设置
随机种子、2～4 块碎片和 0.25×～4× 动画速度。“生成场景”会重建同一个
MuJoCo 场景，并将其俯视相机 RGB 图像送入视觉页面；“识别碎片”和“还原
矩形”会同步触发 MuJoCo 识别及 PIPER-L 逐块拼接。二维碎片动画由机械臂实际
运动进度回传驱动。

命令行运行：

```bash
python3 -m mujoco_sim.run_sim --pieces 4 --seed 7
```

无窗口自动验证：

```bash
MUJOCO_GL=egl python3 -m mujoco_sim.run_sim --pieces 3 --seed 7 --headless
```

初始相机图和最终结果默认保存在 `output/mujoco/`。

## PIPER-L 素材来源与隔离

`assets/piper_l/` 保存了从本机控制站所使用 ROS 描述包复制出的
`piper_l_description.urdf`、动力学版 URDF、7 个 STL 网格，以及控制站使用
的同源 MJCF 基准。仿真运行只读取当前项目中的副本，不访问、不修改也不启动
原机械臂控制站。

## 数据隔离

`scene_builder.py` 生成碎片真值并只输出 MJCF 场景。视觉规划函数
`plan_from_camera_rgb()` 的唯一输入是 MuJoCo 相机渲染的 RGB 图像，不接收
碎片真值位姿、生成顶点或邻接关系。吸附对象通过末端附近的物理距离/接触条件
确定，不按生成编号直接指定。
