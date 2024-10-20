'''
Evaluate the
'''
from glob import glob
import time
import torch
import cv2
import pandas as pd
import numpy as np

from datetime import datetime
import torchvision.transforms as transforms
import os
import sys
sys.path.append('./models')
sys.path.append('./utils')
from seg_utils import merge_boxes_near_boundary
from mAP_utils import calculate_map
import yolov5
from arch import RTSR as  RTSR_train
from mini_arch import RTSR as  mini_arch



def main():
    img_paths = sorted(glob(os.path.expanduser('~/Documents/datasets/Traffic_Aerial_Images_for_Vehicle_Detection/val_1280to320/sec6/LR/*.png')))
    label_paths = sorted(glob(os.path.expanduser('~/Documents/datasets/Traffic_Aerial_Images_for_Vehicle_Detection/val_labels/sec6/*.txt')))

    ###### configuration #######
    ignore_num = 10  ## bypass the first few of frame's prediction latency when calculate the average fps
    device_od = 'cuda'
    model_name_od = 'yolov5'  ##yolov5 , efficientDet, ssd
    device_SR = 'cuda'
    iou_threshold =0.2## calculate mAP
    SR_model = 'standard' ##  'standard', 'mini'

    ###############################

    start1 = time.time()

    SR_loaded = False
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts image to tensor and normalizes to [0, 1]
    ])
    tensor = transforms.ToTensor()

    ##2. load od (yolov5)
    model_od_zoo = ['yolov5', 'ssd', 'efficientDet']
    assert model_name_od in model_od_zoo, f'Model name is not correct, shall be one of {model_od_zoo}'
    if model_name_od == 'efficientDet': import efficientDet  ## do this way cause it conflicts with "import yolov5"
    model_od = eval(model_name_od + '.load_model(device_od)') ## load model in yolov5.py or ssd.py or efficientNet.py

    ### only for cuda
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)  ###Roy  only for cuda

    is_opened = True
    # while (video_capturer.isOpened()):
    dur_list, fps_list ,dur_cuda_list = [], [], []
    map_list = []

    data = pd.DataFrame(
        columns=['latency_whole', 'fps', 'all_cuda_latency'])
    frame_id = 0
    for i, img_path in enumerate(img_paths[1:]):
        frame_id += 1
        label = np.genfromtxt(label_paths[i], delimiter=' ')  ## convert txt to label array
        if label.ndim < 2: label = label[np.newaxis, :] ## if only 1 object annocated, keep the labels in 2 dimension
        if label.size ==0 : continue  ## if there is no annotation
        piece = cv2.imread(img_path)

        if is_opened:
            start = time.time()
            starter.record()  # Roy for cuda


            if SR_model == 'standard':
                if not SR_loaded:
                    SR_net = RTSR_train(sr_rate=4, N=16).to(device_SR)
                    ckpt ='./logs/ckpts/RTSR_N16_epochs180.pth'
                    SR_net.load_state_dict((torch.load(ckpt)))
                    SR_net.eval()
                    SR_loaded = True
                with torch.no_grad():
                    piece_tensor = tensor(piece).unsqueeze(0).to(device_SR)
                    piece = SR_net(piece_tensor).squeeze(0).detach().cpu().numpy()
                    piece = piece.transpose((1, 2, 0))
                    piece = (piece * 255).astype(np.uint8)

            elif SR_model == 'mini':
                if not SR_loaded:
                    SR_net = mini_arch(sr_rate=4, N=8).to(device_SR)
                    ckpt = './logs/ckpts/RTSR_N8_epochs600.pth'
                    SR_net.load_state_dict((torch.load(ckpt)))
                    SR_net.eval()
                    SR_loaded = True
                with torch.no_grad():
                    piece_tensor = tensor(piece).unsqueeze(0).to(device_SR)
                    piece = SR_net(piece_tensor).squeeze(0).detach().cpu().numpy()
                    piece = piece.transpose((1, 2, 0))
                    piece = (piece * 255).astype(np.uint8)

            else:
                print('Model shall be one of: "standard" or "mini"!')

            if model_name_od == 'yolov5':
                vehicle_num, res_yolo = count_vehicle_yolo(model_od, piece, device_od)

                # 获取预测结果
                xywhn_piece = res_yolo.xywhn[0].detach().cpu().numpy()  # [class, confidence, x, y, w, h]
                confidences = xywhn_piece[:, -2]  # 获取置信度
                class_ids = xywhn_piece[:, -1].astype(int)  # 获取类别ID

                xywhn_piece = xywhn_piece[:, [5, 4, 0, 1, 2, 3]]  # [class, confidence, x, y, w, h]

                class_ids[np.logical_or(class_ids == 5, class_ids == 2)] = 0
                class_ids = np.where(class_ids != 0, 1, class_ids)

                xywhn_piece[:, 0] = class_ids
                xywhn_piece[:, 1] = confidences
                all_xywhn = xywhn_piece.copy()

            ender.record()  # Roy for cuda
            torch.cuda.synchronize()  # Roy for cuda
            t3 = starter.elapsed_time(ender) / 1000  # Roy for cuda


            if not all_xywhn.size>0:
                mAP = 0.0
                # print(f'Index {i}: Nothing detected!')
            else:
                map_xywhn = np.delete(np.array(all_xywhn), 1, axis=1)  ## delete confidence columnn
                mAP = calculate_map(map_xywhn, label, num_classes=2,
                                    iou_threshold=iou_threshold)  ## efficientDet 0.1?

            map_list.append(mAP)

            if model_name_od == 'yolov5':
                frame = draw_boxes(piece, all_xywhn)

            dur = time.time() - start
            fps = 1 / dur
            dur_list.append(dur)  ## also called latency_whole
            dur_cuda_list.append(t3)
            fps_list.append(fps)

            cv2.imshow('Images', frame)

            key = cv2.waitKey(1) ###
            if key == 27:  ## ord('q')
                is_opened = False
                break
        else:  ## loop of if is_open
            break


    data.latency_whole = dur_list
    data.fps = fps_list
    data.all_cuda_latency = dur_cuda_list


    device_state = 'odCuda' if device_od == 'cuda' else 'odCpu'

    now = datetime.now() # current date and time
    date_time = now.strftime("%Y%m%d%H%M")

    print(f'fps: {(frame_id-ignore_num)/sum(dur_list[ignore_num:])}.')
    print('mAP:', np.mean(map_list),'.')
    print('Cuda run time:', np.mean(dur_cuda_list[ignore_num:]), '.')
    # print(od_images)
    data.to_csv('./result/' + 'performance_' +model_name_od+'_'+ device_state + '_'+date_time+'.csv')

    # video_capturer.release()
    # cv2.destroyAllWindows()
    print(f'Total runing time is {time.time() - start1} seconds.')

def draw_boxes(image, boxes, colors=None):
    image_copy = image.copy()

    height, width = image.shape[:2]  # Get the dimensions of the image

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    text_scale = 0.5
    text_thickness = 1

    # Define default color if none provided
    if colors is None:
        colors = [(0, 255, 0)] * len(boxes)

    for i, box in enumerate(boxes):
        class_id, confidence, x, y, w, h = box
        x1, y1, x2, y2 = int(x * width - w * width / 2), int(y * height - h * height / 2), int(x * width + w * width / 2), int(y * height + h * height / 2)
        color = colors[i % len(colors)]

        # Draw the bounding box
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, thickness)

        if class_id is not None and confidence is not None:
            label = f"{class_id}: {confidence:.2f}"
            text_size, _ = cv2.getTextSize(label, font, text_scale, text_thickness)
            text_w, text_h = text_size
            cv2.rectangle(image_copy, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
            cv2.putText(image_copy, label, (x1, y1 - 5), font, text_scale, (0, 0, 0), text_thickness, cv2.LINE_AA)

    return image_copy


################ below codes are for couting vehicles ################

def count_vehicle_yolo(model, frame, device):
    yolo_vehicles = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    vehicle_num = 0
    res = yolov5.predict(model, frame)
    pred_objects = (res.pred[0])[:,
                   -1].tolist()  ## result is list of object classes detected, ,. e.g.[2,7] 'res_s2.pred[0]' is tensor, [:,-1] means object classes
    for obj in pred_objects:
        if obj in yolo_vehicles: vehicle_num += 1

    return vehicle_num, res  ## return both vehicle count and result of yolo

if __name__ == '__main__':
    main()