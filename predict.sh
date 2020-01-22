export PYTHONPATH=".:${PYTHONPATH}"
# ===================================
# retinanet_R_50_FPN_3x input imgs dir
# ===================================
# python demo/demo.py \
# 	--config-file configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml \
# 	--input_dir /public/tempShare/testpics \
# 	--output /home/huangzeyu/tmp/output \
# 	--opts MODEL.WEIGHTS detectron2://COCO-Detection/retinanet_R_50_FPN_3x/137849486/model_final_4cafe0.pkl


# ===================================
# retinanet_R_50_FPN_3x input video
# ===================================
# python demo/demo.py \
# 	--config-file configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml \
# 	--video-input /home/huangzeyu/tmp/hello.h264 \
# 	--output /home/huangzeyu/tmp/output \
# 	--opts MODEL.WEIGHTS detectron2://COCO-Detection/retinanet_R_50_FPN_3x/137849486/model_final_4cafe0.pkl


# ===================================
# cascade_mask_rcnn_R_50_FPN_3x input video
# ===================================
# python demo/demo.py \
# 	--config-file configs/Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml \
# 	--video-input /public/tempShare/partvideo.mp4 \
# 	--output /home/huangzeyu/tmp/output \
# 	--opts MODEL.WEIGHTS detectron2://Misc/cascade_mask_rcnn_R_50_FPN_3x/144998488/model_final_480dd8.pkl


# ===================================
# cascade_mask_rcnn_R_50_FPN_3x input video
# ===================================
# python demo/demo.py \
# 	--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# 	--input /public/tempShare/testpics/person_and_bike_125.png \
# 	--output /home/huangzeyu/detectron2/output \
# 	--opts MODEL.WEIGHTS /home/huangzeyu/detectron2/output/model_0004999.pth


# ===================================
# yolov3 input img
# ===================================
python demo/demo.py \
	--config-file projects/Yolov3/configs/yolov3_darknet53_1x.yaml \
	--input_dir /home/huangzeyu/tmp/goggles/datasets/train \
	--output /home/huangzeyu/tmp/goggles/output/train/detected_persons \
	--crop_output /home/huangzeyu/tmp/goggles/output/train/detected_persons \
	--opts MODEL.WEIGHTS weights/yolov3.pth
# python demo/demo.py \
# 	--config-file projects/Yolov3/configs/yolov3_darknet53_1x.yaml \
# 	--input datasets/coco/val2017/000000289594.jpg \
# 	--output output/yolov3 \
# 	--opts MODEL.WEIGHTS projects/Yolov3/yolov3.pth
