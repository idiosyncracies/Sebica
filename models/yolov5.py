import torch
import cv2
import numpy as np
import os

def load_model(device):
    if device == 'cuda' and torch.cuda.is_available():
        device_yolov5s = 0  # device = 0： 'cuda'。 or 'cpu'
    else:
        device_yolov5s = 'cpu'
    local_source = os.path.join(os.environ['HOME'], './.cache/torch/hub/ultralytics_yolov5_master')
    # model = torch.hub.load(local_source, 'yolov5s', source='local', pretrained=True,
    #                        device=device_yolov5s)  ## model is loaded locally
    model = torch.hub.load(local_source,  'custom', path=local_source+'/yolov5s.pt', source='local',
                           device=device_yolov5s)  ## model+pretrained pt file are loaded locally (copy pt file to related path
    model.conf = 0.52  # confidence threshold (0-1) 0.52
    model.iou = 0.45  # NMS IoU threshold (0-1). 0.45
    return model

def predict(model, frame):
    return model([frame])
def display(results):
    return np.squeeze(results.render())

def main():
    root = os.path.join(os.environ['HOME'],'./Documents/datasets/traffic')  #ubuntu
    device ='cuda'
    model = load_model(device)
    video = os.path.join(root, 'Traffic count, monitoring with computer vision. 4K, UHD, HD._4.mp4')  # home
    video_capturer = cv2.VideoCapture(video)
    # fps_video = video_capturer.get(cv2.CAP_PROP_FPS)
    # frame_width = int(video_capturer.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(video_capturer.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    while (video_capturer.isOpened()):
        is_opened, frame = video_capturer.read()  # (720, 1280, 3)， H,W,C (BGR)
        if is_opened:
            results = predict(model, frame)
            frame = display(results)
            cv2.imshow('Result', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        if not is_opened: break

    video_capturer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()