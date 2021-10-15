import torch


class Detector:
    def __init__(self, weight_path, size=320, conf=0.3, iou=0.5):
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
