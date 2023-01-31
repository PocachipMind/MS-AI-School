from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from PIL import Image

# pip install Pillow
# pip install opencv-python==4.5.5.64

cofig_file = "./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"
checkpoint_file = "./faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

model = init_detector(cofig_file, checkpoint_file, device='cuda')
test_img = './demo/demo.jpg'

output = inference_detector(model, test_img)

show_result_pyplot(model, test_img, output)