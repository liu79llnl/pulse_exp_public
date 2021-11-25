import argparse
import cv2
import sys
import numpy as np
import insightface
# from pythonPackage import insightface as insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


parser = argparse.ArgumentParser(description='insightface gender-age test')
# general
parser.add_argument('--ctx', default=0, type=int, help='ctx id, <0 means using cpu')
args = parser.parse_args()

app = FaceAnalysis(allowed_modules=['detection', 'genderage'])
# app.prepare(ctx_id=args.ctx, det_size=(640,640))
app.prepare(ctx_id=args.ctx, det_size=(160, 160))

# img = ins_get_image('t1')
img = cv2.imread("/usr/xtmp/jl888/celeba_test_ref_insightface/39/39_2.jpg")
# print(img)
print(img.shape)
faces = app.get(img)
# assert len(faces)==6
for face in faces:
    print(face.bbox)
    print(face.sex, face.age)

# import onnx
# from onnx2pytorch import ConvertModel

# onnx_model = onnx.load("/home/users/jl888/.insightface/models/antelopev2/genderage.onnx")
# pytorch_model = ConvertModel(onnx_model, debug=True)
# print(pytorch_model)