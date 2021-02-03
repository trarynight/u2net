import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
import collections
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob
import cv2

from data_loader_albu import generate_transforms, SalObjDataset, SalObjDatasetT
from albumentations import (
    Compose,
	SmallestMaxSize,
)

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi) / (ma-mi)
    return dn

def save_output(image_name, pred, d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir + imidx + '.png')

######### modified by wjj for more smooth results generate ###########
# def save_output(image_name, pred, d_dir):
#     img_name = image_name.split(os.sep)[-1]
#     image = cv2.imread(image_name)
#     h, w = image.shape[:2]
#     predict = pred
#     predict = predict.squeeze()
#     predict_np = predict.cpu().data.numpy()
#     predict_np = np.uint8(predict_np * 255)
#     predict_np = cv2.resize(predict_np, (w, h))
#     ret, thresh = cv2.threshold(predict_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     kernel = np.ones((3, 3), np.uint8)
#     opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
#     im = cv2.dilate(opening, kernel, iterations=3) # sure_bg -> im
#     # dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
#     # ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
#     # cv2.imwrite('./sure_fg.jpg', sure_fg)
#     # cv2.imwrite('./sure_bg.jpg', sure_bg)
#     # sure_fg = np.uint8(sure_fg)
#     # unknown = cv2.subtract(sure_bg, sure_fg)
#     # cv2.imwrite('./subtract.jpg', unknown)
#     # ret, markers = cv2.connectedComponents(sure_fg)
#     # markers += 1
#     # markers[unknown == 255] = 0
#     # markers = cv2.watershed(image, markers)
#     # image[markers == -1] = [0, 255, 0]
#     # cv2.imwrite('./res.jpg', image)

#     # _, im = cv2.threshold(predict_np * 255, 100, 255, cv2.THRESH_BINARY)
#     # im = cv2.resize(im, (w, h))
#     # im /= 255
#     # im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
#     # image = np.uint8(image * im)

#     # im = np.uint8(im)
#     im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
#     cv2.bitwise_and(image, im, image)
#     # cv2.imwrite('./res.jpg', image)

#     aaa = img_name.split(".")
#     bbb = aaa[0:-1]
#     imidx = bbb[0]
#     for i in range(1, len(bbb)):
#         imidx = imidx + "." + bbb[i]
#     # cv2.imwrite(d_dir + imidx + 'mask.jpg', predict_np*255)
#     cv2.imwrite(d_dir + imidx + '.jpg', image)
######### modified ended ###########

def main():

    # --------- 1. get image path and name ---------
    model_name = 'u2netp' # u2netp u2net
    data_dir = '/data2/wangjiajie/datasets/scene_segment1023/u2data/'
    image_dir = os.path.join(data_dir, 'test_imgs')
    prediction_dir = os.path.join('./outputs/', model_name + '/')
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir, exist_ok=True)
    # tra_label_dir = 'test_lbls/'

    image_ext = '.jpg'
    # label_ext = '.jpg' # '.png'
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')

    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(f'test img numbers are: {len(img_name_list)}')

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=Compose([SmallestMaxSize(max_size=320),])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if (model_name == 'u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)
    elif(model_name == 'u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 1)
    
    # net.load_state_dict(torch.load(model_dir))
    checkpoint = torch.load(model_dir)
    d = collections.OrderedDict()
    for key, value in checkpoint.items():
        tmp = key[7:]
        d[tmp] = value
    net.load_state_dict(d)
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:", img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7= net(inputs_test)

        # normalization
        pred = 1.0 - d1[:, 0, :, :]
        pred = normPRED(pred)

        # save results to test_results folder
        save_output(img_name_list[i_test], pred, prediction_dir)

        del d1, d2, d3, d4, d5, d6, d7

if __name__ == "__main__":
    main()
