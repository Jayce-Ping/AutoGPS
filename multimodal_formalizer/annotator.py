# This code aims to rearrange the geometry3k and pgps9k dataset
# for a more convenient use in the problem_parser module.
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple
import argparse

# Place all images to the same directory
# name each image by its problem id
# annotate the images with the point labels

def annotate_image(
        point_positions : Dict[str, Tuple[float, float]],
        image_path : str,
        output_path : str,
        **kwargs
    ):
    """
    Annotate the image with the given point position.
    """
    img = Image.open(image_path)
    width, height = img.size
    img_array = np.array(img)
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)

    x_offset = kwargs.get('x_offset', 0)
    y_offset = kwargs.get('y_offset', 0)
    point_size = kwargs.get('point_size', 80)

    if 'x_offset' in kwargs:
        del kwargs['x_offset']
    if 'y_offset' in kwargs:
        del kwargs['y_offset']
    if 'point_size' in kwargs:
        del kwargs['point_size']
    

    ax.imshow(img_array)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    
    for point, pos in point_positions.items():
        x, y = pos
        default_keywork = {
            'fontsize': 36,
            'color': 'blue',
            'ha': 'left',
            'va': 'bottom',
        }
        default_keywork.update(kwargs)
        # Add the point label to the image
        ax.text(x + x_offset, y + y_offset, point, **default_keywork)

        # Draw a red point at the position
        ax.scatter(x + x_offset, y + y_offset, color='red', zorder=5, s=point_size)
    
    ax.axis('off')
    ax.set_title("")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()
    
    output_img = Image.open(output_path)
    if output_img.size != (width, height):
        output_img = output_img.resize((width, height), Image.LANCZOS)
        output_img.save(output_path)


def rearrange_geometry3k(data_dir, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
    annotation_file = os.path.join(data_dir, "geometry3k_test.json")

    annotations = json.load(open(annotation_file, 'r'))

    for problem_id in range(0, 3002):
        if str(problem_id) not in annotations:
            continue

        if problem_id <= 2100:
            phase = 'train'
        elif problem_id <= 2400:
            phase = 'val'
        else:
            phase = 'test'
        
        problem_dir = os.path.join(data_dir, phase, str(problem_id))

        img_path = os.path.join(problem_dir, "img_diagram.png")
        annotated_img_path = os.path.join(problem_dir, "img_diagram_point.png")
        if os.path.exists(annotated_img_path):
            # Copy the annotated image to output_dir
            output_filename = f"{problem_id}_diagram.png"
            output_path = os.path.join(output_dir, output_filename)
            if not os.path.exists(output_path):
                os.system(f"cp {annotated_img_path} {output_path}")
            continue
        # If the annotated image does not exist, use the original image and annotate it
        if os.path.exists(img_path):
            output_filename = f"{problem_id}_diagram.png"
            output_path = os.path.join(output_dir, output_filename)

            if not os.path.exists(output_path):
                annotate_image(
                    annotations[str(problem_id)]['point_positions'],
                    img_path,
                    output_path,
                    fontsize=24,
                    ha='center',
                    va='center'
                    )


def rearrange_pgp9k(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    annotation_file = os.path.join(data_dir, "pgps9k_test.json")

    annotations = json.load(open(annotation_file, 'r'))
    image_dir = os.path.join(data_dir, "Diagram")

    test_file = os.path.join(data_dir, 'PGPS9K', "test.json")
    test_data = json.load(open(test_file, 'r'))
    for pid in test_data.keys():
        problem_id  = pid.replace("prob_", "")
        diagram_name = test_data[pid]['diagram']
        img_path = os.path.join(image_dir, diagram_name)
        # Copy the image to output_dir
        output_filename = f"{problem_id}_diagram.png"
        output_path = os.path.join(output_dir, output_filename)
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        if not os.path.exists(output_path):
            annotate_image(
                annotations[str(problem_id)]['point_positions'],
                img_path,
                output_path,
                fontsize=28
                )


def parse_args():
    parser = argparse.ArgumentParser(description='Parse geometry3k and PGPS9K annotations')
    parser.add_argument('--geometry3k_root', type=str, default='../datasets/geometry3k', help='Root directory of geometry3k dataset')
    parser.add_argument('--pgps9k_root', type=str, default='../datasets/PGPS9K', help='Root directory of PGPS9K dataset')
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    geometry3k_path = args.geometry3k_root
    output_dir = os.path.join(geometry3k_path, "images")
    rearrange_geometry3k(
        geometry3k_path,
        output_dir
    )

    pgps9k_path = args.pgps9k_root
    output_dir = os.path.join(pgps9k_path, "images")
    rearrange_pgp9k(
        pgps9k_path,
        output_dir
    )