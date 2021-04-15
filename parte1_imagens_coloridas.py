import sys
import math
from scipy import misc
from scipy import ndimage
import cv2
import numpy as np
import matplotlib.pyplot as plt
import ntpath

vetor_b = np.array([0.2989, 0.5870, 0.1140])
matriz_a = np.array([[0.393, 0.769, 0.189],
                     [0.394, 0.686, 0.168],
                     [0.272, 0.534, 0.131]])

for image_path in sys.argv[1:]:
    image_name = ntpath.basename(image_path)
    img = cv2.imread(image_path)

    img_a = np.dot(img, matriz_a)
    img_a = img_a.clip(max=[255,255,255])
    img_b = np.tensordot(img, vetor_b, axes=([2], [0]))
    img_b = img_b.clip(max=[255]).reshape(img.shape[0:2])

    cv2.imwrite('imagens_mascaradas_parte1/a_' + image_name, img_a)
    cv2.imwrite('imagens_mascaradas_parte1/b_' + image_name, img_b)
