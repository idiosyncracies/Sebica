'''
This program present the inference of the approach,
single stream
For training, it is in the notebook

'''

import os
import time
import torch
import cv2
import pandas as pd
import numpy as np
import sys
from datetime import datetime
sys.path.append('./models')
sys.path.append('./utils')
import yolov5
from glob import glob
from mAP_utils import calculate_map

def main():
    img_paths = sorted(glob(os.path.expanduser('~/Documents/datasets/Traffic_Aerial_Images_for_Vehicle_Detection/val_1280to320/sec6/GT/*.png')))
    label_paths = sorted(glob(os.path.expanduser('~/Documents/datasets/Traffic_Aerial_Images_for_Vehicle_Detection/val_labels/sec6/*.txt')))

    ###### configuration #######
    ignore_num = 10
    device_od = 'cuda'
    model_name_od = 'yolov5'  ##yolov5 , efficientDet, ssd
    iou_threshold =0.2## calculate mAP

    ###############################

    start1= time.time()
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
        frame_id+=1
        label = np.genfromtxt(label_paths[i], delimiter=' ')
        frame = cv2.imread(img_path)
        # is_opened, frame = video_capturer.read()
        if is_opened:
            start = time.time()

            # set pause to snapshot images in the video
            # key1 = cv2.waitKey(2)
            # if key1 == ord('p'):
            #     while (True):
            #         key2 = cv2.waitKey(5)
            #         if key2 == ord('o'):
            #             break
            ##########################################

            starter.record()  ###Roy for cuda
            if model_name_od == 'yolov5':
                vehicle_num, res_yolo = count_vehicle_yolo(model_od, frame, device_od)
                ## convert predicted boxes in format [[class, x,y,w.h,][...]..]

                ########### calculate mAP ################
                xywhn = res_yolo.xywhn[0].detach().cpu().numpy()
                xywhn[:, [0, 1, 2, 3, 4, 5]] = xywhn[:, [5, 4, 0, 1, 2, 3]]  ## [class, confident, x, y, w,h]
                # xywhn = xywhn[:, :-1]  ## car: 2, bus:5, train:6
                xywhn[:, 0][np.logical_or(xywhn[:, 0] == 5,
                                          xywhn[:, 0] == 2)] = 0  ## replace class 2,5 to 0, so fit the label
                xywhn[:, 0] = np.where(xywhn[:, 0] != 0, 1, xywhn[:, 0])  ## replace all others to 1
                xywhn = np.delete(xywhn, 1, axis=1)
                mAP = calculate_map(xywhn, label, num_classes=2,
                                        iou_threshold=iou_threshold)  ## efficientDet 0.1?
                # print(mAP)
                map_list.append(mAP)  ## pred format [class, confident, x, y, w,h], label format [class, x, y, w,h]
                ############################################
                # print(mAP)

                # res_yolo = yolov5.predict(model_od, frame)  ###Roy, only for teting exclusively run yolo,, exclusive run yolov5 is 56fps
            else:
                pass

            ender.record() ###Roy for cuda
            torch.cuda.synchronize() ###Roy for cuda
            t3= starter.elapsed_time(ender) / 1000  ###Roy for cuda


            if model_name_od == 'yolov5':
                frame = yolov5.display(res_yolo)  ## or: frame = np.squeeze(res_s2.render())

            else:
                pass

            cv2.imshow('Images', frame)

            dur = time.time() - start
            fps = 1 / dur
            dur_list.append(dur)  ## also called latency_whole
            dur_cuda_list.append(t3)
            fps_list.append(fps)

            key = cv2.waitKey(2) ###
            if key == 27:  ## ord('q')
                is_opened = False
                break
        else:  ## loop of if is_open
            break

    ## record and save od_images

    data.latency_whole = dur_list
    data.fps = fps_list
    data.all_cuda_latency = dur_cuda_list


    device_state = 'odCuda' if device_od == 'cuda' else 'odCpu'

    now = datetime.now() # current date and time
    date_time = now.strftime("%Y%m%d%H%M")

    print(f'fps: {(frame_id-ignore_num)/sum(dur_list[ignore_num:])}.')
    print('mAP:', np.mean(map_list),'.')
    print('Cuda run time:', np.mean(dur_cuda_list[ignore_num:]),'.')
    # print(od_images)
    data.to_csv('./result/' + 'performance_' +model_name_od+'_'+ device_state + '_'+date_time+'.csv')

    # video_capturer.release()
    # cv2.destroyAllWindows()

    print(f'Total runing time is {time.time()-start1} seconds.')


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