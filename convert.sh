# ===================================
# yolov3 covnert
# ===================================
# python projects/Yolov3/convert.py \
#     --config-file projects/Yolov3/configs/yolov3_darknet53_1x.yaml \
#     --initial_weights weights/yolov3.weights \
#     --output_dir weights/


# ===================================
# darknet53 covnert
# ===================================
python projects/Darknet/convert.py \
    --config-file projects/Darknet/configs/darknet_3x.yaml \
    --initial_weights weights/darknet53.weights \
    --output_dir weights/