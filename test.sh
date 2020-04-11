export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=".:${PYTHONPATH}"
# ===================================
# maskrcnn test
# ===================================
# python tools/train_net.py \
#     --num-gpus 3 \
#     --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
#     --eval-only \
#     --resume \
#     MODEL.WEIGHTS /home/huangzeyu/detectron2/weights/model_final_a54504.pkl


# ===================================
# tridentnet test
# ===================================
# python projects/TridentNet/train_net.py \
#     --num-gpus 3 \
#     --config-file projects/TridentNet/configs/tridentnet_fast_R_50_C4_3x.yaml \
#     --eval-only \
#     --resume


# ===================================
# yolov3 test
# ===================================
python projects/Yolov3/train_net.py \
    --num-gpus 1 \
    --config-file projects/Yolov3/configs/yolov3_1x.yaml \
    --resume \
    --eval-only \
    MODEL.WEIGHTS weights/yolov3.pth


# ===================================
# resnet50 test
# ===================================
# python projects/Resnet/train_net.py \
#     --num-gpus 2 \
#     --config-file projects/Resnet/configs/resnet50.yaml \
#     --eval-only \
#     MODEL.WEIGHTS weights/resnet50-19c8e357.pth



# ===================================
# resnet101 test
# ===================================
# python projects/Resnet/train_net.py \
#     --num-gpus 2 \
#     --config-file projects/Resnet/configs/resnet101.yaml \
#     --eval-only \
#     MODEL.WEIGHTS weights/resnet101-5d3b4d8f.pth


# ===================================
# resnet152 test
# ===================================
# python projects/Resnet/train_net.py \
#     --num-gpus 2 \
#     --config-file projects/Resnet/configs/resnet152.yaml \
#     --eval-only \
#     MODEL.WEIGHTS weights/resnet152-b121ed2d.pth


# ===================================
# darknet53 test
# ===================================
# python projects/SmokeCall/train_net.py \
#     --num-gpus 4 \
#     --eval-only \
#     --resume \
#     --config-file projects/Backbone/configs/darknet_4x.yaml \
#     MODEL.WEIGHTS weights/darknet53.pth

# ===================================
# smoke test
# ===================================
# python projects/SmokeCall/train_net.py \
#     --num-gpus 4 \
#     --eval-only \
#     --resume \
#     --config-file projects/SmokeCall/smoke_call_1x.yaml

# ===================================
# yolov3 input img
# ===================================
# python demo/demo.py \
# 	--is_test 1 \
# 	--config-file projects/Yolov3/configs/yolov3_darknet53_1x.yaml \
# 	--input_dir /home/huangzeyu/tmp/goggles/datasets/test \
# 	--output /home/huangzeyu/tmp/goggles/output/test/drawn \
# 	--crop_output /home/huangzeyu/tmp/goggles/output/test/persons \
# 	--opts MODEL.WEIGHTS weights/yolov3.pth
	
