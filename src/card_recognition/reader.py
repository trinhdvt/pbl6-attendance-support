import time
from collections import defaultdict
from typing import List

import torch
import yaml
from PIL.Image import Image
from loguru import logger
from vietocr.tool.translate import build_model, translate, process_input


class Reader:
    def __init__(self, cfg_path, weight_path):
        config = self.load_config(cfg_path)
        device = config['device']
        #
        model, vocab = build_model(config)
        #
        model.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))
        #
        self.config = config
        self.model = model
        self.vocab = vocab
        self.device = device

    @staticmethod
    def load_config(path):
        with open(path, encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _process_input(self, img):
        return process_input(img, self.config['dataset']['image_height'],
                             self.config['dataset']['image_min_width'],
                             self.config['dataset']['image_max_width'])

    def predict(self, image: Image, show_time=False) -> str:
        """
        Transformer single predict

        :param image: PIL image to predict
        :param show_time: True to show predicted time
        :return: Predicted result in string format
        """
        start = time.time()

        # preprocess
        img = self._process_input(image)
        img = img.to(self.device)

        # feedforward
        sequence, _ = translate(img, self.model)

        # decode
        sequence = self.vocab.decode(sequence[0].tolist())

        #
        if show_time:
            logger.debug(f'Predicted in {time.time() - start}')
        return sequence

    def batch_predict(self, images: List[Image]) -> List[str]:
        """
        Transformer batch predict

        :param images: List of PIL images to predicted
        :return: List of predicted result in string format
        """

        #
        batch = defaultdict(list)
        batch_idx = defaultdict(list)
        batch_pred = {}
        results_seq = [""] * len(images)

        # create batch
        for i, img in enumerate(images):
            img = self._process_input(img)

            batch[img.shape[-1]].append(img)
            batch_idx[img.shape[-1]].append(i)

        # feedforward then decode
        for k, batch_item in batch.items():
            batch_k = torch.cat(batch_item, 0).to(self.device)
            seq, _ = translate(batch_k, self.model)
            seq = seq.tolist()
            seq = self.vocab.batch_decode(seq)

            batch_pred[k] = seq

        # retrieve result
        for k in batch_pred:
            idx = batch_idx[k]
            seq = batch_pred[k]
            for i, j in enumerate(idx):
                results_seq[j] = seq[i]
        #
        return results_seq
