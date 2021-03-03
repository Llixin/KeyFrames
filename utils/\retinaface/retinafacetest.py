from __future__ import print_function
import os
import cv2
import argparse
import shutil
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from .data.config import cfg_mnet, cfg_re50
from .layers.functions.prior_box import PriorBox
from .utils.nms.py_cpu_nms import py_cpu_nms
from .models.retinaface import RetinaFace
from .utils.box_utils import decode, decode_landm
from .utils.timer import Timer

workpath = os.path.split(os.path.realpath(__file__))[0]


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    #     print('Missing keys:{}'.format(len(missing_keys)))
    #     print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    #     print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    #     print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    #     print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def test(
        data,
        index,
        save_folder=workpath + '/eval',
        save_image=False,
        trained_model=workpath + '/weights/Resnet50_Final.pth',
        network='resnet50',
        cpu=False,
        confidence_threshold=0.02,
        top_k=5000,
        nms_threshold=0.4,
        keep_top_k=750,
        vis_thres=0.99):
    torch.set_grad_enabled(False)
    cfg = None
    if network == "mobile0.25":
        cfg = cfg_mnet
    elif network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, trained_model, cpu)
    net.eval()
    #     print('Finished loading model!')
    # print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if cpu else "cuda")
    net = net.to(device)

    # save file
    if save_image:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        fw = open(os.path.join(save_folder, '_dets.txt'), 'w')

    # testing dataset
    test_dataset = dict()
    for idx in index:
        test_dataset[idx] = data[idx]
    num_images = len(test_dataset)

    # testing scale
    resize = 1

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    imgresultpath = workpath + "/rf_results/"
    if save_image:
        if os.path.exists(imgresultpath):
            shutil.rmtree(imgresultpath)
        os.makedirs(imgresultpath)

    result = dict()

    # testing begin
    for i, img in enumerate(list(test_dataset.values())):
        img = np.float32(img)
        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        _t['forward_pass'].tic()
        loc, conf, landms = net(img)  # forward pass
        _t['forward_pass'].toc()
        _t['misc'].tic()
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        # order = scores.argsort()[::-1][:top_k]
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)

        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        # dets = dets[:keep_top_k, :]
        # landms = landms[:keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        _t['misc'].toc()

        # save dets
        if save_image:
            fw.write('{:s}\n'.format(i))
            fw.write('{:d}\n'.format(dets.shape[0]))
            for k in range(dets.shape[0]):
                xmin = dets[k, 0]
                ymin = dets[k, 1]
                xmax = dets[k, 2]
                ymax = dets[k, 3]
                score = dets[k, 4]
                w = xmax - xmin + 1
                h = ymax - ymin + 1
                # fw.write('{:.3f} {:.3f} {:.3f} {:.3f} {:.10f}\n'.format(xmin, ymin, w, h, score))
                fw.write('{:d} {:d} {:d} {:d} {:.10f}\n'.format(int(xmin), int(ymin), int(w), int(h), score))
                # print('{:s}: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(img_name, i + 1, num_images, _t['forward_pass'].average_time, _t['misc'].average_time))

        result[index[i]] = list()
        # show image

        for b in dets:
            # b[0:14]
            # 0,1,2,3：人脸矩形坐标
            # 4：人脸概率
            # 5,6,7,8,9,10,11,12,13,14：5个特征点坐标
            if b[4] < vis_thres:
                continue
            result[index[i]].append(b)
            if save_image:
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)  # 红色
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)  # 黄色
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)  # 紫色
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)  # 绿色
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)  # 蓝色
        # save image
        if save_image:
            name = imgresultpath + i + ".jpg"
            cv2.imwrite(name, img_raw)

    if save_image:
        fw.close()

    return result


if __name__ == "__main__":
    pass
