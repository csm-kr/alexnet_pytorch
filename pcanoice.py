import numpy as np
from PIL import Image


class PCANoisePIL(object):
    def __init__(self,
                 alphastd=0.1,
                 eigval=np.array([1.148, 4.794, 55.46]),
                 eigvec=np.array([[-0.5675, 0.7192, 0.4009],
                                  [-0.5808, -0.0045, -0.8140],
                                  [-0.5836, -0.6948, 0.4203],])
                 ):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec
        self.set_alpha()

    def __call__(self, img):

        # 1. pil to numpy
        img_np = np.array(img)                                   # [H, W, C]
        offset = np.dot(self.eigvec * self.alpha, self.eigval)
        img_np = img_np + offset
        img_np = np.maximum(np.minimum(img_np, 255.0), 0.0)
        ret = Image.fromarray(np.uint8(img_np))
        return ret

    def set_alpha(self, ):
        # change per each epoch
        self.alpha = np.random.normal(0, self.alphastd, size=(3,))