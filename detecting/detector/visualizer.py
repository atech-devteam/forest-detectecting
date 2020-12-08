import colorsys
import random
import matplotlib.pyplot as plt 
import matplotlib
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
from skimage.measure import find_contours
import numpy as np
import cv2

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    print(colors)
    return colors

gentle_grey = (45, 65, 79)
white = (255, 255, 255)

def display_instances_with_cv2(image, boxes, masks, class_ids, class_names, scores, colors, font_size=1):
    N = boxes.shape[0]
    
    if N:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # Convert to Grayscale background
    masked_image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    for i in range(N):
        class_id = class_ids[i]
        color = colors[class_id-1]

        # Bounding box
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]
        camera_color = (color[0] * 255, color[1] * 255, color[2] * 255)
        cv2.rectangle(masked_image, (x1, y1), (x2, y2), camera_color, 1)

        # Mask
        mask = masks[:, :, i]
        alpha = 0.3
        for c in range(3):
            masked_image[:, :, c] = np.where(mask == 1,
                                             image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                             masked_image[:, :, c])

        # Label
        score = scores[i]
        label = class_names[class_id]
        caption = '{} {:.2f}'.format(label, score) if score else label

        # Get caption text size
        ret, baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)

        # Put the rectangle and text on the top left corner of the bounding box
        # cv2.rectangle(masked_image, (x1, y1), (x1 + ret[0], y1 + ret[1] + baseline), camera_color, -1)
        blk = np.zeros(masked_image.shape, np.uint8)
        cv2.rectangle(blk, (x1, y1), (x1 + ret[0], y1 + ret[1] + baseline), camera_color, cv2.FILLED)
        masked_image = cv2.addWeighted(masked_image, 1.0, blk, 0.25, 1)
        # putText(image, text, org, font, scale, color, thickness.. )
        cv2.putText(masked_image, caption, (x1, y1 + ret[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, white, 1, lineType=cv2.LINE_AA)


        # Put the rectangle and text on the bottom left corner
        # cv2.rectangle(masked_image, (x1, y2 - ret[1] - baseline), (x1 + ret[0], y2), camera_color, -1)
        # cv2.putText(masked_image, caption, (x1, y2 - baseline),
                    # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return masked_image.astype(np.uint8)
    # return masked_image

def display_instances(image, boxes, masks, class_ids, class_names,
                    scores=None, title="",
                    figsize=(16, 16), ax=None,
                    show_mask=True, show_bbox=True,
                    colors=None, captions=None, outputFileName=None, dpi=200, own_colors=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
        """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        asprat = float(image.shape[0]) / image.shape[1]
        fig, ax = plt.subplots(1, figsize=(figsize[0], figsize[1] * asprat))
        auto_show = True

    # Generate random colors
    # If you want to use random colors instead of list of color according class_ids, uncomment the line following this line.
    # colors = colors or self.random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.axis('off')
    ax.axis('off')
    fig.tight_layout(pad=0)
    ax.margins(0)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        if own_colors:
            color = own_colors[class_ids[i]]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    
    
    
    matplotlib.use('Agg') 
    ax.imshow(masked_image.astype(np.uint8))
    fig.canvas.draw()
    cvImg = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    cvImg  = cvImg.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    cvImg = cv2.cvtColor(cvImg,cv2.COLOR_RGB2BGR)
    if fig:
        plt.close(fig)

    if outputFileName:
        cv2.imwrite(outputFileName, cvImg)
    else:
        return cvImg

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                image[:, :, c] *
                                (1 - alpha) + alpha * color[c] * 255,
                                image[:, :, c])
    return image