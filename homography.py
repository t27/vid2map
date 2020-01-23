from collections import defaultdict
import numpy as np
import cv2


global_trajectories = defaultdict(list)

frame_wise_info = {}
base_map = cv2.imread("./TopView.png")
raw_map = cv2.imread("./TopView_Map.png")


fourcc = cv2.VideoWriter_fourcc(*"MP4V")
hm_vid = cv2.VideoWriter("map_view2.mp4", fourcc, 30.0, (1920, 1080))

fourcc = cv2.VideoWriter_fourcc(*"MP4V")
raw_vid = cv2.VideoWriter("raw_map_view.mp4", fourcc, 30.0, (1920, 1080))


def get_pick_color(model):
    model_id = id(model)
    r = int(model_id / (256 ** 3))
    g = int(model_id % (256 ** 3) / (256 ** 2))
    b = int(model_id % (256 ** 2) / 256)
    a = int(model_id % 256)
    return (r, g, b, a)


def homography(x, y):
    H2to1_ = np.array(
        [
            [-1.36144871e01, 6.36629902e00, 1.20790173e04],
            [-1.61253076e00, 1.74034924e00, 1.92856470e03],
            [-2.45761937e-03, 6.52584268e-03, 1.00000000e00],
        ]
    )
    out = np.matmul(np.linalg.inv(H2to1_), np.array([[x], [y], [1]]))
    out = out / out[2]
    return out[0][0], out[1][0]


def renderhomography(frame_result, colors):
    global global_trajectories
    for res in frame_result:
        x, y, obj_id, frame_id = res

        x_map, y_map = homography(x, y)
        if x_map < 0 or y_map < 0:
            print("Negative err: ", x, y)
        # frame_wise_info[frame_id] = (x_map, y_map, obj_id)

        # global_trajectories[obj_id].append((y_map, x_map))
        global_trajectories[obj_id].append((x_map, y_map))

        global_trajectories = clean_global_traj(global_trajectories)

        showTrajectories(colors)


def clean_global_traj(global_trajectories):
    # if we have K objects and the length of the path of K points is less than L, remove that id from the dict
    K = 30
    L = 15
    new_glob = defaultdict(list)
    for id in global_trajectories:
        if len(global_trajectories[id]) < K:
            new_glob[id] = global_trajectories[id]
        else:
            length = 0
            for i in range(1, len(global_trajectories[id]) - 1):
                length += np.linalg.norm(
                    np.array(global_trajectories[id][i])
                    - np.array(global_trajectories[id][i - 1])
                )
            if length > L:
                new_glob[id] = global_trajectories[id]
    return new_glob


def showTrajectories(colors):
    # draw lines on base_map
    base_map = cv2.imread("./TopView.png")
    raw_map = cv2.imread("./TopView_Map.png")

    for id in global_trajectories:
        color = colors[int(id) % len(colors)]
        color = [i * 255 for i in color]
        # print(color)
        # print(global_trajectories[id])
        cv2.polylines(
            base_map,
            np.array([global_trajectories[id]], dtype=np.int32),
            False,
            color,
            thickness=5,
        )
        cv2.polylines(
            raw_map,
            np.array([global_trajectories[id]], dtype=np.int32),
            False,
            color,
            thickness=5,
        )
    hm_vid.write(base_map)
    raw_vid.write(raw_map)
    # cv2.imshow("MAP", base_map)
