import cv2
import torch
import math
from NeuralImplicit import NeuralImplicit
from OverfitShapes import Renderer, Camera
from FileUtilities import parallelForEachFile

def train(filename):
    mesh = NeuralImplicit()
    mesh.encode(filename)
    return mesh

#mesh_folder = "./" #PATH TO FOLDER CONTAINING MESH FILES
#parallelForEachFile(mesh_folder, train, [".obj",".stl"], 4)
cube = NeuralImplicit()
cube.lr = 1e-3
cube.adaptive_lr = True
cube.encode("cube.obj", num_samples=1100000, oversample_ratio=20)
cube.model.to(torch.device("cpu"))
renderer = Renderer(*cube.renderable())
def rotate_vec(vec, angle):
    rad = math.radians(angle)
    return (vec[0]*math.cos(rad) - vec[1]*math.sin(rad),
            vec[0]*math.sin(rad) + vec[1]*math.cos(rad),
            vec[2])

def dir_vec(vec):
    mag = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])
    return (-1*vec[0] / mag, -1*vec[1] / mag, -1*vec[2] / mag)

num_frames = 60
angle_step = (2 * math.pi) / num_frames
curr_angle = 0
cam = Camera()
frames = []
for i in range(num_frames):
    curr_angle += angle_step
    pos = (math.sqrt(3)*math.cos(curr_angle), math.sqrt(3)*math.sin(curr_angle), 1)
    print(pos)
    cam.position = pos
    cam.direction = dir_vec(pos)
    cam.side = (0,1,0)
    renderer.camera = cam
    frames += [renderer.render()]

while (True):
    for frame in frames:
        cv2.imshow("img", frame)
        key = cv2.waitKey(int(2.0/num_frames*1000))
        if (key == 'q'):
            exit()