import numpy as np


class Features:
    def __init__(self, OCR=None, ratio=None, HOG=None  ):
        self.OCR = OCR
        self.ratio = ratio
        self.HOG = HOG

    