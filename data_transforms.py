import cv2
import math
import random
import numpy as np
import torch
import transforms3d


class Compose(object):
    def __init__(self, transforms):
        self.transformers = []
        for tr in transforms:
            transformer = eval(tr['callback'])
            parameters = tr['parameters'] if 'parameters' in tr else None
            self.transformers.append({
                'callback': transformer(parameters),
                'objects': tr['objects']
            })  # yapf: disable

    def __call__(self, data):
        for tr in self.transformers:
            transform = tr['callback']
            objects = tr['objects']
            rnd_value = np.random.uniform(0, 1)
            scare_value = np.random.randint(80, 120) * 0.01
            for k, v in data.items():
                if k in objects and k in data:
                    if transform.__class__ in [ScalePoints]:
                        data[k] = transform(v, rnd_value, scare_value)
                    elif transform.__class__ in [RandomRotatePoints, RandomMirrorPoints]:
                        data[k] = transform(v, rnd_value)
                    else:
                        data[k] = transform(v)

        return data


class ToTensor(object):
    def __init__(self, parameters):
        pass

    def __call__(self, arr):
        shape = arr.shape
        if len(shape) == 3:    # RGB/Depth Images
            arr = arr.transpose(2, 0, 1)

        # Ref: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/2
        return torch.from_numpy(arr.copy()).float()


class RandomRotatePoints(object):
    def __init__(self, parameters):
        pass

    def __call__(self, ptcloud, rnd_value):
        trfm_mat = transforms3d.zooms.zfdir2mat(1)
        if rnd_value <= 0.5:
            scale = 3.0
            angle = random.uniform(-math.pi, math.pi) * scale / 180.0
            trfm_mat = np.dot(transforms3d.axangles.axangle2mat([0,1,0], angle), trfm_mat)
        ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)
        return ptcloud


class ScalePoints(object):
    def __init__(self, parameters):
        pass

    def __call__(self, ptcloud, rnd_value, scale, translation=True):
        if rnd_value <= 0.5:
            ptcloud = ptcloud * scale
            
        return ptcloud
  
        
class RandomMirrorPoints(object):
    def __init__(self, parameters):
        pass

    def __call__(self, ptcloud, rnd_value):
        trfm_mat = transforms3d.zooms.zfdir2mat(1)
        trfm_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
        trfm_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)
        if rnd_value <= 0.25:
            trfm_mat = np.dot(trfm_mat_x, trfm_mat)
            trfm_mat = np.dot(trfm_mat_z, trfm_mat)
        elif rnd_value > 0.25 and rnd_value <= 0.5:  # lgtm [py/redundant-comparison]
            trfm_mat = np.dot(trfm_mat_x, trfm_mat)
        elif rnd_value > 0.5 and rnd_value <= 0.75:
            trfm_mat = np.dot(trfm_mat_z, trfm_mat)
        ptcloud = np.dot(ptcloud, trfm_mat.T)
        return ptcloud