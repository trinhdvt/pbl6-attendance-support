from typing import Dict, List, Optional

import torch
from PIL.Image import Image


class Detector:
    def __init__(self, weight_path, size=416, conf=0.4, iou=0.5):
        self.model = torch.hub.load("ultralytics/yolov5", 'custom', path=weight_path)
        self.model.conf = conf
        self.model.iou = iou
        self.size = size
        self.class_names = self.model.model.names

    def detect_crop(self, image: Image, save_file=False, verbose=False) -> Dict[str, Optional[Image]]:
        """
        Detect and crop information from image then save to a dict

        :param save_file: True to save detect image
        :param image: PIL Image
        :param verbose: True to show processing time
        :return: Dictionary with key is the class's name and value is the cropped image
        """

        # inference
        results = self.model(image, size=self.size)

        # print process step
        if verbose:
            results.print()
        if save_file:
            results.save()

        # get bounding box from detected results
        bounding_box = self.get_bounding_box(results, image)

        # crop image for each class
        return self.crop(image, bounding_box)

    def get_bounding_box(self, detect_rs, img: Image) -> Dict[str, Optional[List[int]]]:
        """
        Get bounding box for each class from detected results

        :param detect_rs: Detect results from YOLOV5 model
        :param img: Original image
        :return: An dictionary: Key [label] -> Bounding_box [x_min, y_min, x_max, y_max] or None
        """

        detect_rs = detect_rs.xyxyn[0].tolist()
        """
             xmin    ymin    xmax   ymax  confidence  class
        0  749.50   43.50  1148.0  704.5    0.874023      0
        1  433.50  433.50   517.5  714.5    0.687988      1
        """

        #
        width, height = img.size
        bounding_box = {}

        # get bounding box and concat them for each class
        for row in detect_rs:
            #
            label = self.class_names[int(row[5])]
            if label not in bounding_box.keys():
                bounding_box[label] = [width + 100, height + 100, 0, 0]

            #
            bbox = row[:4]
            current_row = bounding_box[label]
            current_row[0] = min(int(width * bbox[0]), current_row[0])
            current_row[1] = min(int(height * bbox[1]), current_row[1])
            current_row[2] = max(int(width * bbox[2]), current_row[2])
            current_row[3] = max(int(height * bbox[3]), current_row[3])

        return bounding_box

    def crop(self, img: Image, bounding_box: Dict[str, List[int]]) -> Dict[str, Optional[Image]]:
        """
        Crop image for each class

        :param img: Original image
        :param bounding_box: Bounding box of each class
        :return: A dictionary: Key [label] -> Value [PIL Image] or None
        """

        crop_rs = {
            label: None for label in self.class_names
        }
        padding = [-5, -2, 5, 2]
        for label, bbox in bounding_box.items():
            if len(bbox) != 4:
                continue

            # adding some padding
            # not padding with class name
            if label != "class":
                bbox = list(map(lambda x, y: x + y, padding, bbox))
            crop_rs[label] = img.crop(bbox)
            # crop_rs[label].save(f"{log_dir}/{label}.png")

        return crop_rs
