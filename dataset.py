import torch
import numpy as np
import torch.utils.data as data
import h5py
import os
import data_transforms


class ShapeNetH5(data.Dataset):
    def __init__(self, train=True, npoints=2048, novel_input=True, novel_input_only=False):
        if train:
            self.input_path = './data/mvp_train_input.h5'
            self.gt_path = './data/mvp_train_gt_%dpts.h5' % npoints
        else:
            self.input_path = './data/mvp_test_input.h5'
            self.gt_path = './data/mvp_test_gt_%dpts.h5' % npoints
        self.npoints = npoints
        self.train = train

        input_file = h5py.File(self.input_path, 'r')
        self.input_data = np.array((input_file['incomplete_pcds'][()]))
        self.labels = np.array((input_file['labels'][()]))
        self.novel_input_data = np.array((input_file['novel_incomplete_pcds'][()]))
        self.novel_labels = np.array((input_file['novel_labels'][()]))
        input_file.close()

        gt_file = h5py.File(self.gt_path, 'r')
        self.gt_data = np.array((gt_file['complete_pcds'][()]))
        self.novel_gt_data = np.array((gt_file['novel_complete_pcds'][()]))
        gt_file.close()

        if novel_input_only:
            self.input_data = self.novel_input_data
            self.gt_data = self.novel_gt_data
            self.labels = self.novel_labels
        elif novel_input:
            self.input_data = np.concatenate((self.input_data, self.novel_input_data), axis=0)
            self.gt_data = np.concatenate((self.gt_data, self.novel_gt_data), axis=0)
            self.labels = np.concatenate((self.labels, self.novel_labels), axis=0)

        print(self.input_data.shape)
        print(self.gt_data.shape)
        print(self.labels.shape)
        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len
    
    def _get_transforms(self, train):
        if train:
            return data_transforms.Compose([{
                'callback': 'RandomMirrorPoints',
                'objects': ['partial_cloud', 'gtcloud']
            },{
                'callback': 'ScalePoints',
                'objects': ['partial_cloud', 'gtcloud']
            },{
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        else:
            return data_transforms.Compose([{
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
    
    def __getitem__(self, index):
        transforms = self._get_transforms(self.train)
        permutation = torch.randperm(2048)
        partial = self.input_data[index]
        partial = partial[permutation]
        complete = self.gt_data[index // 26]
        label = (self.labels[index])
        data = {}
        data['partial_cloud'] = partial
        data['gtcloud'] = complete
        data = transforms(data)
        data['label'] = label

        return data



