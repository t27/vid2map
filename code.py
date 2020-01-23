import detectron2
from detectron2.utils.logger import setup_logger
import sort

setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
from tqdm import tqdm

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import time
import matplotlib.pyplot as plt

from homography import renderhomography

_tracker = sort.Sort()

relevant_classes = [0, 1, 2, 3, 5, 7, 9, 11]

SHOW_IMAGE = True

fourcc = cv2.VideoWriter_fourcc(*"MP4V")
vid = cv2.VideoWriter("video_view.mp4", fourcc, 30.0, (1920, 1080))


classes = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
# init code

DETECTRON_MODELS = {
    "mrcnn1": [
        "./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl",
    ],
    "mrcnn2": [
        "./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
        "detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl",
    ],
    "frcnn1": [
        "./detectron2_repo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
        "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl",
    ],
    "frcnn2": [
        "./detectron2_repo/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl",
    ],
}


def getPredictor():
    cfg = get_cfg()
    detector = "frcnn1"
    cfg.merge_from_file(DETECTRON_MODELS[detector][0])
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = DETECTRON_MODELS[detector][1]
    predictor = DefaultPredictor(cfg)
    return cfg, predictor


pbar = tqdm(total=100000)


def run():
    cfg, predictor = getPredictor()
    cap = cv2.VideoCapture("test.mp4")
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    if cap.isOpened() == False:
        print("Error opening video stream or file")
    f_no = 0
    while cap.isOpened():
        # Capture frame-by-frame
        t1 = time.time()
        ret, frame = cap.read()
        if f_no % 2 != 0:
            continue

        frame_result = []  # stores x,y,id,frameid

        if ret is True:
            pbar.update(1)
            t2 = time.time()

            # Run predictor model
            outputs = predictor(frame)

            # extract the predictor model
            detection_classes = (
                outputs["instances"].pred_classes.to("cpu").unsqueeze(1).numpy()
            )
            detection_boxes = outputs["instances"].pred_boxes.tensor.to("cpu").numpy()
            detection_scores = (
                outputs["instances"].scores.to("cpu").unsqueeze(1).numpy()
            )
            tr_boxes = np.hstack(
                (detection_boxes, detection_scores, detection_scores, detection_classes)
            )

            tr_boxes = np.array([x for x in tr_boxes if x[6] in relevant_classes])
            tr_boxes[:, 6] = 2
            tracked_boxes = _tracker.update(tr_boxes)
            unique_labels = np.unique(detection_classes)
            # n_cls_preds = len(unique_labels)

            for x1, y1, x2, y2, obj_id, cls_pred in tracked_boxes:
                box_h = int(y2 - y1)
                box_w = int(x2 - x1)
                x1 = int(x1)
                y1 = int(y1)
                color = colors[int(obj_id) % len(colors)]
                color = [i * 255 for i in color]

                cls = classes[int(cls_pred)]
                cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), color, 4)
                cv2.rectangle(
                    frame, (x1, y1 - 35), (x1 + len(cls) * 19 + 60, y1), color, -1
                )
                cv2.putText(
                    frame,
                    cls + "-" + str(int(obj_id)),
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    3,
                )
                frame_result.append([((x2 + x1) / 2), y2, int(obj_id), f_no / 2])

            t3 = time.time()
            # print(
            #     "Capture Time:", t2 - t1,
            # )
            # print("Processing Time:", t3 - t2)
            ## Call homography and show map

            renderhomography(frame_result, colors)

            # We can use `Visualizer` to draw the predictions on the image.
            # v = Visualizer(
            #     frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2
            # )
            # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            # # cv2_imshow(v.get_image()[:, :, ::-1])
            # img = v.get_image()[:, :, ::-1]
            if SHOW_IMAGE:
                # im = cv2.resize(frame, (640, 480))
                vid.write(frame)
                # cv2.imshow("Frame", frame)
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
