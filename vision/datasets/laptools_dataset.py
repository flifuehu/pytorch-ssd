import numpy as np
import pathlib
import xml.etree.ElementTree as ET
import cv2
import pandas as pd
import os
import torch


class LapToolsDataset:

    def __init__(self, root, dataset_csv, transform=None, target_transform=None, is_test=False, keep_difficult=False):
        """Dataset for LapTools data.
        Args:
            root: the folder where images are actually stored
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.is_test = is_test
        self.ids = pd.read_csv(dataset_csv, header=None,
                               names=['frame', 'x1', 'y1', 'x2', 'y2', 'label', 'fold', 'img_h', 'img_w'])
        self.keep_difficult = keep_difficult

        classes = pd.read_csv(os.path.join(os.path.dirname(dataset_csv), 'class_mapping.txt'), header=None)
        classes.loc[-1] = ['BACKGROUND', 0]
        classes.index = classes.index + 1
        classes.sort_index(inplace=True)
        self.class_names = classes[0].values
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        boxes, labels, is_difficult = self._get_annotation(index)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(index)
        # self.show(image, boxes, labels)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        # self.show(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        # self.show(image, boxes, labels)
        return image, boxes, labels

    def get_image(self, index):
        # image_id = self.ids[index]
        image = self._read_image(index)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids.loc[index, 'frame']
        return image_id, self._get_annotation(index)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, index):

        image_id = self.ids.iloc[index]

        # get all the annotations for the 'index' image
        target = self.ids[self.ids['frame'] == image_id['frame']]
        boxes = target.loc[:, ['x1', 'y1', 'x2', 'y2']].values
        labels = [self.class_dict[r['label']] for _, r in target.iterrows()]
        is_difficult = np.zeros_like(labels)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, index):
        image_file = os.path.join(self.root, self.ids.iloc[index]['frame'])
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def show(self, img, boxes, labels):
        import matplotlib.pyplot as plt
        import matplotlib.patches as pat
        fig, ax = plt.subplots(1)
        for obj in boxes:
            obj = obj.astype(np.int)
            ax.add_patch(pat.Rectangle((obj[0], obj[1]), (obj[2]-obj[0]), (obj[3]-obj[1]), edgecolor=(1,0,0), fill=False))
        if isinstance(img, torch.Tensor):
            img = img.numpy().transpose((1,2,0))
        plt.imshow(img)
        plt.title(labels)
        plt.show()



