# coding:utf-8
import argparse
import collections
import json
import sys

import cv2
import numpy as np
import os
import pandas as pd
import scenedetect
import shutil
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import warnings
from PIL import Image
from models.model import *
from scenedetect.detectors import ContentDetector
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.video_manager import VideoManager
from tqdm import tqdm
from utils.bdlaplacian.blur import blur_detection as blurtest
from utils.retinaface.retinafacetest import test as facetest


class KeyFrames:

    def __init__(self, args=None, gpu_id=None):
        self.work_path = os.path.split(os.path.realpath(__file__))[0]

        if args is None:
            parser = argparse.ArgumentParser()
            parser.add_argument('--model', type=str, default=self.work_path + "/checkpoints/epoch-82.pth",
                                help='path to pretrained model')
            parser.add_argument('--test_images', type=str, default=self.work_path + "/images/",
                                help='path to folder containing images')
            parser.add_argument('--out', type=str, default=self.work_path + "/result_img",
                                help='save the result images')
            parser.add_argument('-n', '--number', type=int, default=5, help='how many results do you want to save')
            parser.add_argument('--cpu', action="store_true", default=False)
            args = parser.parse_args()
        self.args = args

        if gpu_id is not None and isinstance(gpu_id, int):
            assert gpu_id < torch.cuda.device_count(), f'参数 gpu_id {gpu_id} 越界，没有那么多GPU。'
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        if args.cpu:
            self.device = torch.device("cpu")
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                warnings.warn('GPU不可用，将使用CPU。')
                self.device = torch.device("cpu")

        base_model = models.vgg16(pretrained=True)

        model = NIMA(base_model)

        try:
            model.load_state_dict(torch.load(args.model))
            # print('successfully loaded model')
        except IOError:
            print('Model failed to load model')

        self.model = model.to(self.device)

        self.model.eval()

        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

    @staticmethod
    def get_index(result: list):
        shot_id = list()
        for res in result:
            shot_id.append(res['frame_index'])
        return shot_id

    @staticmethod
    def save_img(imgs: list, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        for i, st in enumerate(imgs):
            if isinstance(st, list) and len(st) > 1:
                for j, im in enumerate(st):
                    name = path + '/' + str(i) + '_' + str(j) + '.jpg'
                    # print(name)
                    cv2.imwrite(name, im)
            else:
                if isinstance(st, list):
                    st = st[0]
                name = path + '/' + str(i) + '.jpg'
                # print(name)
                cv2.imwrite(name, st)

    @staticmethod
    def process_video(video_path: str, json_path: str, shot=False):
        video_manager = VideoManager([video_path])
        # video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager = SceneManager(StatsManager())
        scene_manager.add_detector(ContentDetector())
        print('processing the video...')
        frames = list()
        while True:
            ret, img = video_manager.read()
            if not ret:
                break
            frames.append(img)

        # ###### shot #######
        shots = list()
        if shot:
            video_manager._started = False
            video_manager.reset()
            video_manager.start()
            scene_manager.detect_scenes(frame_source=video_manager)
            shot_list = scene_manager.get_scene_list(video_manager.get_base_timecode())
            for i, shot in enumerate(shot_list):
                shots.append((shot[0].get_frames(), shot[1].get_frames()))

        # ###### story #######
        with open(json_path, 'r', encoding='utf8') as f:
            data = json.load(f)
        stories = list()
        for story in data:
            story['inpoint'] //= 400000
            story['outpoint'] //= 400000
            stories.append((story['inpoint'], story['outpoint']))

        return frames, shots, stories

    def process_shot(self, images: list, n=0):
        args = self.args
        if n:
            args.number = n

        if os.path.exists(args.out):
            shutil.rmtree(args.out)
        os.makedirs(args.out)

        index_nima = dict()

        for i, img in enumerate(images):
            mean = 0.0
            im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # im = im.convert('RGB')
            imt = self.test_transform(im)
            imt = imt.unsqueeze(dim=0)
            imt = imt.to(self.device)
            with torch.no_grad():
                out = self.model(imt)
            out = out.view(10, 1)
            for j, e in enumerate(out, 1):
                mean += j * e
            index_nima[i] = float(mean)

        ######################
        sort_result = sorted(index_nima.items(), key=lambda item: item[1], reverse=True)
        top_result = sort_result[:min(args.number, len(sort_result))]
        index_nima = dict(top_result)

        ######################
        # print('======face detecting======')
        face_d = facetest(data=images, index=list(index_nima.keys()))
        """face_d:
        dict：{ img_index: list() }
        若img无人脸，对应的list()为空
        list()的每一行[0:14]表示一个人脸，其中：
        [0,1,2,3]：人脸矩形坐标
        [4]：人脸概率
        [5,6,7,8,9,10,11,12,13,14]：5个特征点坐标
        """
        result = list()

        for img, faces in list(face_d.items()):
            res = dict()
            res["frame_index"] = int(img)
            res["nima_confidence"] = index_nima[int(img)]

            if not faces:
                res["exist_face"] = None
                # name = os.path.join(args.out, str(img) + '.jpg')
                # cv2.imwrite(name, images[int(img)])
            else:
                # 若人脸面积小于阈值，则忽略
                mainFace = list()
                height, width, _ = images[0].shape
                face_threshold = height * width // 100
                for i in range(len(faces)):
                    face = faces[i]
                    area = (face[2] - face[0]) * (face[3] - face[1])
                    if area < face_threshold:
                        continue
                    mainFace.append(face)

                res_face = list()
                # ##===侧脸===###
                for i in range(len(mainFace)):
                    face_status = dict()
                    face = mainFace[i]
                    if min(face[5], face[11]) <= face[9] <= max(face[7], face[13]):
                        face_status["is_front"] = True
                    else:
                        face_status["is_front"] = False
                    res_face.append(face_status)

                # ##===闭眼===###
                ed = cv2.CascadeClassifier(self.work_path + '/checkpoints/haarcascade_eye_tree_eyeglasses.xml')
                for i in range(len(mainFace)):
                    face = mainFace[i]
                    face = images[int(img)][int(face[1]):int(face[3]), int(face[0]):int(face[2])]
                    eyes = ed.detectMultiScale(face, 1.01, 1)
                    if len(eyes) > 0:
                        res_face[i]["eye_opened"] = True
                    else:
                        res_face[i]["eye_opened"] = False

                res["exist_face"] = res_face

            # 若存在闭眼侧脸，则淘汰
            valid = True
            if res["exist_face"]:
                for face in res["exist_face"]:
                    if face["is_front"] is False or face["eye_opened"] is False:
                        valid = False
                        break
            if valid or not result:
                result.append(res)

        ######################
        # blur_d = blurtest(images=images, index=list(index_nima.keys()))
        # if len(blur_d) > 2:
        #     maxvarimg, maxvar = 0, 0
        #     minvarimg, minvar = 0, float('inf')
        #     for line in blur_d:
        #         if line[1] > maxvar:
        #             maxvarimg, maxvar = line[0], line[1]
        #         if line[1] < minvar:
        #             minvarimg, minvar = line[0], line[1]
        #     blur_d.pop(blur_d.index((maxvarimg, maxvar)))
        #     blur_d.pop(blur_d.index((minvarimg, minvar)))
        # mean = 0.0
        # for line in blur_d:
        #     var = line[1]
        #     mean += var / len(blur_d)
        # variance = 0.0
        # for line in blur_d:
        #     var = line[1]
        #     variance += (var - mean) * (var - mean) / len(blur_d)
        # std = np.sqrt(variance)
        # # 标准差过大，有图片模糊
        # if mean > 0 and std / mean > 0.1:
        #     delete = set()
        #     for line in blur_d:
        #         img, var = line[0], line[1]
        #         if var < mean:
        #             delete.add(img)
        #     del_idx = list()
        #     for i in range(len(result)):
        #         dic = result[i]
        #         if dic['frame_index'] in delete:
        #             del_idx.append(i)
        #     for i in del_idx[::-1]:
        #         if len(result) > 1:
        #             result.pop(i)

        return result

    def predict(self, video_path: str, json_path: str):
        """
        对每个story切分成shot，对每个shot提取代表帧，再从这些代表帧里提取story的代表帧
        :param video_path: 视频文件路径
        :param json_path: 视频的story切分点的json文件路径
        :return:
        """
        # 得到划分好的story，shot
        frames, shots, stories = self.process_video(video_path, json_path, shot=True)
        # print('frames:', len(frames))
        # print('shots:', shots)
        # print('stories:', stories)
        # 把每个shot放到对应的story里
        story_shot = [list() for _ in range(len(stories))]
        i, flag = 0, [0] + [end for _, end in stories]
        for (sh_begin, sh_end) in shots:
            if sh_end <= flag[i + 1]:
                story_shot[i].append((sh_begin, sh_end))
            else:
                i += 1
        # print('flag:', flag)
        # print('story_shot:', story_shot)
        # 得到每个shot的代表帧
        story_kf = [list() for _ in range(len(stories))]
        story_im = [list() for _ in range(len(stories))]
        for i, story in enumerate(story_shot):
            for j, shot in enumerate(story_shot[i]):
                shot_im = frames[shot[0]:shot[1]]
                story_im[i].append(shot_im)
                story_kf[i].append(self.get_index(self.process_shot(shot_im, 1)))
        # print('story_kf:', story_kf)
        # 把每个story里的所有shot的代表帧整合到一起
        story_kf_im = [list() for _ in range(len(stories))]
        for i in range(len(story_kf)):
            for j in range(len(story_kf[i])):
                for k in story_kf[i][j]:
                    story_kf_im[i].append(story_im[i][j][k])
        # self.save_img(story_kf_im, "video/gdxw/shot/")  # 保存每个shot的keyframes
        # 得到每个story的代表帧
        keyframes = [list() for _ in range(len(stories))]
        for i in range(len(story_kf_im)):
            keyframes[i] = self.get_index(self.process_shot(story_kf_im[i], 1))
        keyframes_im = [list() for _ in range(len(stories))]
        for i in range(len(keyframes)):
            for j in keyframes[i]:
                keyframes_im[i].append(story_kf_im[i][j])
        return keyframes_im

    def predict2(self, video_path: str, json_path: str):
        """
        对每个story不再切分shot，直接提取代表帧
        :param video_path: 视频文件路径
        :param json_path: 视频的story切分点的json文件路径
        :return:
        """
        frames, _, stories = self.process_video(video_path, json_path)
        keyframes = list()
        stories_im = list()
        d = 5  # 去除story首尾的若干帧
        for a, b in stories:
            stories_im.append(frames[a + d:b - d])
            keyframes.append(self.get_index(self.process_shot(frames[a + d:b - d], 1)))
        keyframes_im = list()
        for i, k in enumerate(keyframes):
            keyframes_im.append(stories_im[i][k[0]])
        return keyframes_im


def main():
    path = "video/story/"
    video_path = list()
    json_path = list()
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.mp4':
            video_path.append(path + file)
            json_path.append(path + file.split('.')[0] + '.json')
    test = KeyFrames()
    for video, json in (zip(video_path, json_path)):
        print("========={}, {}=========".format(video, json))
        kf_im = test.predict2(video, json)
        test.save_img(kf_im, video.split('.')[0])


if __name__ == "__main__":
    main()
