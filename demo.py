import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from models import EfficientDet
from torchvision import transforms
import numpy as np
import skimage
from datasets import get_augumentation, VOC_CLASSES
from timeit import default_timer as timer
import argparse
import copy
import numpy as np
from utils import vis_bbox, EFFICIENTDET
# import onnx
# import onnxruntime
import time
# from torch2trt import torch2trt
parser = argparse.ArgumentParser(description='EfficientDet')

parser.add_argument('-n', '--network', default='efficientdet-d0',
                    help='efficientdet-[d0, d1, ..]')
parser.add_argument('-s', '--score', default=True,
                    action="store_true", help='Show score')
parser.add_argument('-t', '--threshold', default=0.6,
                    type=float, help='Visualization threshold')
parser.add_argument('-it', '--iou_threshold', default=0.6,
                    type=float, help='Visualization threshold')
parser.add_argument('-w', '--weight', default='./weights/voc0712.pth',
                    type=str, help='Weight model path')
parser.add_argument('-c', '--cam',
                    action="store_true", help='Use camera')
parser.add_argument('-f', '--file_name', default='pic.jpg',
                    help='Image path')
parser.add_argument('--num_class', default=21, type=int,
                    help='Number of class used in model')
args = parser.parse_args()


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
class Detect(object):
    """
        dir_name: Folder or image_file
    """

    def __init__(self, weights, num_class=21, network='efficientdet-d0', size_image=(512, 512),use_tensorrt=False):
        super(Detect,  self).__init__()
        self.weights = weights
        self.size_image = size_image
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else 'cpu')
        self.transform = get_augumentation(phase='test')
        if(self.weights is not None):
            print('Load pretrained Model')
            checkpoint = torch.load(
                self.weights, map_location=lambda storage, loc: storage)
            params = checkpoint['parser']
            num_class = params.num_class
            network = params.network

        self.model = EfficientDet(num_classes=num_class,
                                  network=network,
                                  W_bifpn=EFFICIENTDET[network]['W_bifpn'],
                                  D_bifpn=EFFICIENTDET[network]['D_bifpn'],
                                  D_class=EFFICIENTDET[network]['D_class'],
                                  is_training=False
                                  )

        if(self.weights is not None):
            state_dict = checkpoint['state_dict']
            self.model.load_state_dict(state_dict)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        self._tensror_rt= use_tensorrt

        if use_tensorrt :
            x = torch.randn(1,3, 512, 512, requires_grad=True)
            self.model.backbone.set_swish(memory_efficient=False)
            if torch.cuda.is_available():
                x=x.cuda()
            self.model= torch2trt(self.model,[x])
            # # Export the model
            # batch_size=1
            # x = torch.randn(batch_size, 3, 512, 512, requires_grad=True)
            # # x = self.transform(to_x)
            # torch.onnx.export(self.model,               # model being run
                  # x,                         # model input (or a tuple for multiple inputs)
                  # "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
                  # export_params=True,        # store the trained parameter weights inside the model file
                  # opset_version=10,          # the ONNX version to export the model to
                  # do_constant_folding=True,  # whether to execute constant folding for optimization
                  # input_names = ['input'],   # the model's input names
                  # output_names = ['output'], # the model's output names
                  # dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                # 'output' : {0 : 'batch_size'}})
            # onnx_model = onnx.load("super_resolution.onnx")
            # onnx.checker.check_model(onnx_model)

            # self._ort_session = onnxruntime.InferenceSession("super_resolution.onnx")


    def _onnx_mdl(self,x):
        # compute ONNX Runtime output prediction
        print(self._ort_session.get_inputs())
        ort_inputs = {self._ort_session.get_inputs()[0].name: to_numpy(x)}
        return self._ort_session.run(None, ort_inputs)


    def process(self, file_name=None, img=None, show=False):
        if file_name is not None:
            img = cv2.imread(file_name)
        origin_img = copy.deepcopy(img)
        augmentation = self.transform(image=img)
        img = augmentation['image']
        img = img.to(self.device)
        img = img.unsqueeze(0)
        print(img.shape)
        with torch.no_grad():
            # ftime= []
            # for _ in range(100):
                # st= time.time()
            scores, classification, transformed_anchors = self.model(img)
                # scores, classification, transformed_anchors = self.model(img)
                # time_pass= time.time()-st
                # ftime.append(time_pass)
            # print("forward time pytorch {} device {}".format(np.mean(ftime),self.device))
            # a=self._onnx_mdl(img)
            bboxes = list()
            labels = list()
            bbox_scores = list()
            colors = list()
            for j in range(scores.shape[0]):
                bbox = transformed_anchors[[j], :][0].data.cpu().numpy()
                x1 = int(bbox[0]*origin_img.shape[1]/self.size_image[1])
                y1 = int(bbox[1]*origin_img.shape[0]/self.size_image[0])
                x2 = int(bbox[2]*origin_img.shape[1]/self.size_image[1])
                y2 = int(bbox[3]*origin_img.shape[0]/self.size_image[0])
                bboxes.append([x1, y1, x2, y2])
                label_name = VOC_CLASSES[int(classification[[j]])]
                labels.append(label_name)

                if(args.cam):
                    cv2.rectangle(origin_img, (x1, y1),
                                  (x2, y2), (179, 255, 179), 2, 1)
                if args.score:
                    score = np.around(
                        scores[[j]].cpu().numpy(), decimals=2) * 100
                    if(args.cam):
                        labelSize, baseLine = cv2.getTextSize('{} {}'.format(
                            label_name, int(score)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(
                            origin_img, (x1, y1-labelSize[1]), (x1+labelSize[0], y1+baseLine), (223, 128, 255), cv2.FILLED)
                        cv2.putText(
                            origin_img, '{} {}'.format(label_name, int(score)),
                            (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 0), 2
                        )
                    bbox_scores.append(int(score))
                else:
                    if(args.cam):
                        labelSize, baseLine = cv2.getTextSize('{}'.format(
                            label_name), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(
                            origin_img, (x1, y1-labelSize[1]), (x1+labelSize[0], y1+baseLine), (0, 102, 255), cv2.FILLED)
                        cv2.putText(
                            origin_img, '{} {}'.format(label_name, int(score)),
                            (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 0), 2
                        )
            if show:
                fig, ax = vis_bbox(img=origin_img, bbox=bboxes,
                                   label=labels, score=bbox_scores)
                fig.savefig('./docs/demo.png')
                plt.show()
            else:
                return origin_img

    def camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Unable to open camera")
            exit(-1)
        count_tfps = 1
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()
        while True:
            res, img = cap.read()
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1

            if accum_time > 1:
                accum_time = accum_time - 1
                fps = curr_fps
                curr_fps = 0
            if res:
                show_image = self.process(img=img)
                cv2.putText(
                    show_image, "FPS: " + str(fps), (10,  20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (204, 51, 51), 2
                )

                cv2.imshow("Detection", show_image)
                k = cv2.waitKey(1)
                if k == 27:
                    break
            else:
                print("Unable to read image")
                exit(-1)
            count_tfps += 1
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    detect = Detect(weights=args.weight)
    print('cam: ', args.cam)
    if args.cam:
        detect.camera()
    else:
        detect.process(file_name=args.file_name, show=True)
