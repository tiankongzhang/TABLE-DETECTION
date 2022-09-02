import json
import matplotlib.pyplot as plt
import cv2
import glob
import os

import numpy as np

from spine import Spine

json.encoder.FLOAT_REPR = lambda x: format(x, '.3f')


def _get_match_box(box, boxes, max_trans=60):
    """

    Args:
        box: (4, 2)
        boxes:  (n, 4, 2)
        max_trans:

    Returns:

    """
    box = np.array(box)
    boxes = np.array(boxes)

    center_x, center_y = np.mean(box[:, 0]), np.mean(box[:, 1])
    center_xs, center_ys = np.mean(boxes[:, :, 0], axis=1), np.mean(boxes[:, :, 1], axis=1)
    closest_index = np.argmin(np.square(center_xs - center_x) + np.square(center_ys - center_y))
    closest_box = boxes[closest_index]
    if np.abs(np.mean(closest_box[:, 0]) - center_x) < max_trans and np.abs(np.mean(closest_box[:, 1]) - center_y) < max_trans:
        return closest_box
    return None


def _process(pred_boxes, gt_boxes):
    """

    Args:
        pred_boxes: (n1, 4, 2)
        gt_boxes: (n2, 4, 2)

    Returns:

    """
    diff_angles = []
    diff_count = len(pred_boxes) - len(gt_boxes)
    for gt_box in gt_boxes:
        gt_spine = Spine(rect=gt_box)
        pred_box = _get_match_box(gt_box, pred_boxes)
        if pred_box is not None:
            pred_spine = Spine(rect=pred_box)
            diff_angle = gt_spine.inner_angle_with(pred_spine)
            diff_angles.append(diff_angle)
        else:
            diff_angles.append(None)
    diff_angles = np.array(diff_angles, dtype=np.float32)
    return diff_angles, diff_count


def _read_txt_as_ndarray(path, scale=1.):
    """

    Args:
        path: str

    Returns:
        boxes: (n, 4, 2)

    """
    boxes = []
    with open(path) as f:
        for line in f.readlines():
            coordinates = list(map(int, line.split(',')[:8]))
            box = np.array(list(zip(coordinates[0::2], coordinates[1::2])))
            boxes.append(np.round(box * scale, 2))
    return np.array(boxes)


def _read_json_as_ndarray(path, scale=1.):
    """

    Args:
        path: str

    Returns:
        boxes: (n, 4, 2)

    """
    boxes = []
    with open(path) as f:
        data = json.load(f)
        for k, v in data.items():
            coordinate = v['coordinate']
            points = []
            for p in coordinate:
                x = coordinate[p]['x']
                y = coordinate[p]['y']
                points.append((x, y))
            box = np.array(points)
            boxes.append(box * scale)
    return np.array(boxes)


def draw_rect(image, box, color):
    spine = Spine(rect=box)
    box = np.array([
        spine.lt, spine.rt, spine.rb, spine.lb
    ])
    cv2.polylines(image, [box.astype(np.int32).reshape((-1, 1, 2))], True, color=color, thickness=1)


def evaluate(pred_dir, test_dir):
    test_image_paths = glob.glob(os.path.join(test_dir, '*.jpg'))
    results = {}
    ignore = ['0012']
    for test_image_path in test_image_paths:
        test_image_name = os.path.basename(test_image_path)
        image_id = test_image_name.split('.')[0]
        if image_id in ignore:
            continue

        test_image = cv2.imread(test_image_path)
        test_h, test_w, _ = test_image.shape

        pred_image_path = os.path.join(pred_dir, test_image_name)
        pred_image = cv2.imread(pred_image_path)
        pred_h, pred_w, _ = pred_image.shape

        gt_boxes = _read_json_as_ndarray(os.path.join(test_dir, '{}.json'.format(image_id)), scale=float(pred_h / test_h))
        pred_boxes = _read_txt_as_ndarray(os.path.join(pred_dir, '{}.txt'.format(image_id)))
        gt_boxes = np.array(sorted(gt_boxes, key=lambda box: np.mean(box[:, 1])))
        diff_angles, diff_count = _process(pred_boxes=pred_boxes, gt_boxes=gt_boxes)

        for box1, box2 in zip(pred_boxes, gt_boxes):
            draw_rect(pred_image, box1, color=(255, 255, 0))
            draw_rect(pred_image, box2, color=(0, 0, 255))

        for angle, box in zip(diff_angles, gt_boxes):
            cv2.putText(pred_image,
                        '%.3f' % angle if not np.isnan(angle) else 'nan',
                        (int(np.mean(box[:, 0])), int(np.mean(box[:, 1]))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        .5,
                        (0, 255, 0),
                        2)
        cv2.imwrite(os.path.join(pred_dir, image_id + '_compare.jpg'), pred_image)

        results[test_image_name] = {
            'diff_count': diff_count,
            'diff_angles': diff_angles,
            'mean_angle_diff': np.nanmean(diff_angles),
            'max_angle_diff': np.nanmax(diff_angles),
            'min_angle_diff': np.nanmin(diff_angles),
        }
    return results


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred_dir', default='outputs/', required=False)
    parser.add_argument('-t', '--test_dir', default='jizhu_test_data/', required=False)
    args = parser.parse_args()
    eval_results = evaluate(pred_dir=args.pred_dir, test_dir=args.test_dir)
    NumpyEncoder.FLOAT_REPR = lambda o: format(o, '.2f')
    result_str = json.dumps(eval_results, indent=2, cls=NumpyEncoder)
    print(result_str)
    mean_angle_diffs = []
    max_angle_diffs = []
    min_angle_diffs = []
    diff_counts = []
    miss_count = {}
    exceed_count = {}

    for key, result in eval_results.items():
        mean_angle_diffs.append(result['mean_angle_diff'])
        max_angle_diffs.append(result['max_angle_diff'])
        min_angle_diffs.append(result['min_angle_diff'])
        diff_counts.append(np.abs(result['diff_count']))
        if result['diff_count'] < 0:
            miss_count[key] = result['diff_count']
        elif result['diff_count'] > 0:
            exceed_count[key] = result['diff_count']
    print('mean_angle_diff: {:.3f}'.format(np.mean(mean_angle_diffs)))
    print('min_angle_diff: {:.3f}'.format(np.mean(min_angle_diffs)))
    print('max_angle_diff: {:.3f}'.format(np.mean(max_angle_diffs)))
    print('diff_count: {:.3f}'.format(np.mean(diff_counts)))
    print('miss_count: {}'.format(miss_count))
    print('exceed_count: {}'.format(exceed_count))
    with open(os.path.join(args.pred_dir, 'results.txt'), 'w') as f:
        f.write(result_str)
