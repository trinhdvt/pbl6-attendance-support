import torch
from PIL import Image


class Detector:
    def __init__(self, weight_path, size=320, conf=0.5, iou=0.5):
        self.model = torch.hub.load("ultralytics/yolov5", 'custom', path=weight_path)
        self.model.conf = conf
        self.model.iou = iou
        self.size = size
        self.class_names = self.model.model.names

    def predict(self, imgs):
        """
        Detect information from image

        :param imgs: PIL Image
        :return: Detected result
        """

        #
        results = self.model(imgs, size=self.size)

        #
        final_results = [[
            [
                {
                    "class": int(pred[5]),
                    "class_name": self.class_names[int(pred[5])],
                    "normalized_box": pred[:4].tolist(),
                    "confidence": float(pred[4])
                }
            ] for pred in result
        ] for result in results.xyxyn]

        #
        return final_results

    def detect_crop(self, image, save_crop=False, verbose=False):
        """
        Detect and crop information from image then save to a dict

        :param save_crop: True to save cropped images
        :param image: PIL Image
        :param verbose: True to show processing time
        :return: Dictionary with key is the class's name and value is the cropped image
        """

        #
        results = self.model(image, size=self.size)

        #
        if verbose:
            results.print()

        # print(results.pandas().xyxy[0])

        #
        cropped = results.crop(save=save_crop)

        # post-processing
        detect_rs = {k: [] for k in self.class_names}

        for rs in cropped:
            # label format: <label_name> <probability_value>
            label = rs['label'].split(' ')[0]

            # box format: list of tensor [x_min, y_min, x_max, y_max]
            box = rs['box']
            box = list(map(lambda x: x.item(), box))  # list[tensor] -> list[float]
            x_min = box[0]  # get x_min to sort detected images in same class

            # image format: numpy array
            im = rs['im'][..., ::-1]  # reshape to HxWxC

            #
            detect_rs[label].append((x_min, im))

        # one class may have several results, sort them by x_min
        # results format: key -> list[PIL image]
        for k, v in detect_rs.items():
            detect_rs[k] = sorted(v, key=lambda x: x[0])
            detect_rs[k] = list(map(lambda x: Image.fromarray(x[1], 'RGB'), detect_rs[k]))

        #
        return detect_rs
