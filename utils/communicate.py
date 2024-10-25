import time
import os
import csv
import re
import numpy as np
import random

def date_time():
    s_l = time.localtime(time.time())
    dt = time.strftime("%Y%m%d", s_l)
    tm = time.strftime("%H%M%S", s_l)
    # print(date, tm )
    return dt, tm

def logger_by_date(col, data_in_row, dir, file_name_prefix):  # save in csv format
    data, _ = date_time()
    file_path = os.path.join(dir,file_name_prefix+ data+'.csv')
    if os.path.exists(file_path):
        with open(file_path, 'a', newline='') as csvf:
            writer =  csv.writer(csvf)
            writer.writerow(data_in_row)
    else:
        with open(file_path, 'w', newline='') as csvf:
            writer =  csv.writer(csvf)
            writer.writerow(col)
            writer.writerow(data_in_row)

def fmt_to_stmp(tm,*arg):
    fmt= "%Y-%m-%d %H:%M:%S" if not arg else arg[0]
    dt, ms =tm.split('.')  # to process with milli-second
    timeArray =time.strptime(dt, fmt)
    timeStamp = time.mktime(timeArray)+int(ms)/1000
    return timeStamp

def save_log(df_data,file_name):
    dt, tm = date_time()
    log_dir = '../result/log/' + dt
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    log_file = file_name + '_'+ dt + tm
    df_data.to_csv(os.path.join(log_dir,log_file)+ '.csv')

def dict2str(dictionary, seperator = ','):
    assert len(seperator) == 1, "Seperator must be type of character"
    string =''
    for key, value in dictionary.items():
        string += str(key)+':'+str(value)
        string += seperator
    return string [:-1]

def str2dict(string, separator=','):
    assert len(separator) == 1, "Separator must be a single character"
    arg = {}
    # Split the string by the separator, but keep tuples together
    arg_list = re.split(r',(?![^()]*\))', string)

    for s in arg_list:
        key, val = s.split(':')
        if re.match(r'\(.*\)', val):
            val = tuple(map(int, re.findall(r'\d+', val)))  # Convert to tuple of integers
        elif re.match(r'^[0-9]+$', val):
            val = int(val)
        arg[key] = val
    return arg

# function for colorizing a segmentation label image:
def label_img_to_color(img):
    label_to_color = {
        0: [128, 64,128],
        1: [244, 35,232],
        2: [ 70, 70, 70],
        3: [102,102,156],
        4: [190,153,153],
        5: [153,153,153],
        6: [250,170, 30],
        7: [220,220,  0],
        8: [107,142, 35],
        9: [152,251,152],
        10: [ 70,130,180],
        11: [220, 20, 60],
        12: [255,  0,  0],
        13: [  0,  0,142],
        14: [  0,  0, 70],
        15: [  0, 60,100],
        16: [  0, 80,100],
        17: [  0,  0,230],
        18: [119, 11, 32],
        19: [81,  0, 81]
        }

    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]

            img_color[row, col] = np.array(label_to_color[label])

    return img_color

def visualize_seg(imgs, outputs):  ## refer:  https://github.com/fregu856/deeplabv3/blob/master/visualization/run_on_seq.py
    outputs = outputs['out'].cpu().numpy()  # (shape: (batch_size, num_classes, img_h, img_w))
    pred_label_imgs = np.argmax(outputs, axis=1)  # (shape: (batch_size, img_h, img_w))
    pred_label_imgs = pred_label_imgs.astype(np.uint8)

    for i in range(pred_label_imgs.shape[0]):
        pred_label_img = pred_label_imgs[i]  # (shape: (img_h, img_w))
        # img_id = img_ids[i]
        img = imgs[i]  # (shape: (3, img_h, img_w))

        img = img.data.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  # (shape: (img_h, img_w, 3))
        img = img * np.array([0.229, 0.224, 0.225])
        img = img + np.array([0.485, 0.456, 0.406])
        img = img * 255.0
        img = img.astype(np.uint8)

        pred_label_img_color = label_img_to_color(pred_label_img)
        overlayed_img = 0.35 * img + 0.65 * pred_label_img_color
        overlayed_img = overlayed_img.astype(np.uint8)

        return overlayed_img


def dataset_split(dataset, split_rate=(0.6, 0.2, 0.2), seed=42):
    random.seed(seed)
    a, b, c = split_rate
    length = len(dataset)
    total = np.arange(length).tolist()
    split = int(a * length)
    train_idx = sorted(random.sample(total, split))
    remain = sorted(set(total) - set(train_idx))
    val_idx = sorted(random.sample(remain, int(b * length)))
    test_idx = sorted(set(remain) - set(val_idx))
    trn = dict.fromkeys(train_idx, 'train')
    val = dict.fromkeys(val_idx, 'val')
    tst = dict.fromkeys(test_idx, 'test')
    trn.update(val)
    trn.update(tst)
    idx_mode = sorted(trn.items())
    train_dataset, val_dataset, test_dataset = [], [], []
    for idx, mode in idx_mode:  # type of idx_mode is tuple NOT dict
        if mode == 'train':
            train_dataset.append(dataset[idx])
        elif mode == 'val':
            val_dataset.append(dataset[idx])
        else: # mode == 'test'
            test_dataset.append(dataset[idx])

    return train_dataset, val_dataset, test_dataset

