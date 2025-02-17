## <center>基于YOLOv8的车辆跟踪、车速计算和车辆统计应用</center>

## 1、环境搭建
 通过conda创建一个python≥3.8环境，激活环境后安装`ultralytics=8.2`、`python-opencv`、`shapely>=2.0.0`:
```bash
conda create -n yolov8 python=3.10
conda activate yolov8
pip install ultralytics==8.2
pip install python-opencv
pip install shapely>=2.0.0
```
**注意项:**
 - `ultralytics`的版本在`v8.3.x`中更新了`SpeedEstimator和ObjectCounter`，因此确保`ultralytics`的版本为`8.2.x`
 - 如果提示Pytorch相关报错，请按照要求安装`PyTorch>=1.8`

## 2、代码

```python
from ultralytics import YOLO
from ultralytics.solutions import speed_estimation,object_counter
import cv2

# 加载官方提供的YOLOv8模型
model = YOLO("yolov8n.pt")
# 获取模型中的对象名称
names = model.model.names
# 打开视频
cap = cv2.VideoCapture("videoplayback.mp4")
assert cap.isOpened(), "Error reading video file"
# 获取视频的宽度、高度和帧率
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# 创建视频写入器，用于输出处理后的视频
video_writer = cv2.VideoWriter("out.avi",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (w, h))

# 设置测速线段的两个端点,一条直线,(x,y)
line_pts = [(0, 180), (640, 180)]
# 初始化速度估计器
speed_obj = speed_estimation.SpeedEstimator()
# 设置速度估计器的参数，包括测速线段、对象名称和是否实时显示图像
# 计数区域或线。只有出现在指定区域内或穿过指定线的对象才会被计数。
speed_obj.set_args(reg_pts=line_pts,
                    names=names,
                    view_img=True)

# 初始化计数器
counter_obj = object_counter.ObjectCounter()
counter_obj.set_args(reg_pts = line_pts,
                        classes_names = names,
                        view_img = False)

# 循环读取视频帧
while cap.isOpened():
    # 读取一帧
    success, im0 = cap.read()
    # 如果读取失败，则退出循环
    if not success:
        break
    tracks = model.track(im0, persist=True, show=False)
    im0 = counter_obj.start_counting(im0,tracks)
    im0 = speed_obj.estimate_speed(im0,tracks)
    # 将处理的结果保存为视频
    # video_writer.write(im0)

# 释放视频读取器和写入器
cap.release()
video_writer.release()
# 销毁所有OpenCV窗口
cv2.destroyAllWindows()
```
## 3、运行结果
![在这里插入图片描述](access/002.jpeg#pic_center)
## 4、`ultralytics=8.3.x中的实现方式`
#### 4.1主要的变化
  - Model类将在解决方案中实例化，因此用户只需要提供模型文件路径。
  - 对象跟踪现在将在解决方案中处理，而不是由外部视频处理循环管理。
  - `yolov10`、`yolov11`版本模型也可以用
#### 4.2 ultralytics=8.3.x版本实现代码

```python
import cv2
from ultralytics import solutions

cap = cv2.VideoCapture("videoplayback.mp4")

assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter("speed_estimation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

speed = solutions.SpeedEstimator(
    model="yolov8n.pt",
    #速度计算区域
    region= [(0, 160), (640, 160), (640, 180), (0, 180)],
    show=True
)

while cap.isOpened():
    success, im0 = cap.read()
    if success:
        out = speed.estimate_speed(im0)
        # video_writer.write(im0)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue
    print("Video frame is empty or video processing has been successfully completed.")
    break

cap.release()
cv2.destroyAllWindows()
```
#### 4.3 ultralytics=8.3.x版本结果
![在这里插入图片描述](access/003.jpeg#pic_center)

