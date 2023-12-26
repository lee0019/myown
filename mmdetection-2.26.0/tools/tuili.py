# 导入库
from mmdet.apis import init_detector, inference_detector
import mmcv

# 选择配置文件以及权重文件，该文件位于之前指定的训练目录work_dirs/yolox_l下
config_file = 'E://mmlab//mmdetection-2.26.0//configs//faster_rcnn//my_faster_rcnn_r50_fpn_2x_coco.py'
checkpoint_file = 'E://mmlab//mmdetection-2.26.0//tools//work_dirs//50epoch_rsod//faster_rcnn//latest.pth'

# 从配置文件和权重文件构建模型，使用0号GPU进行推理
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 指定预测图片
img = 'aircraft_27.jpg'
# 也可以使用下述方法，二者取其一
#img = mmcv.imread('data/infer/new.jpg' )
# 获得推理结果
result = inference_detector(model, img)
# 结果可视化（窗口模式）
model.show_result(img, result)
# 结果可视化（保存图片模式）
model.show_result(img, result, out_file='tui.jpg')
