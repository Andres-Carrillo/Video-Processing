from PyQt5.QtGui import QImage
import numpy as np
import cv2
import itertools
from functools import reduce
from operator import mul
import math


COCO_CLASSES = {0: "person",
  1: "bicycle",
  2: "car",
  3: "motorcycle",
  4: "airplane",
  5: "bus",
  6: "train",
  7: "truck",
  8: "boat",
  9: "traffic light",
  10: "fire hydrant",
  11: "stop sign",
  12: "parking meter",
  13: "bench",
  14: "bird",
  15: "cat",
  16: "dog",
  17: "horse",
  18: "sheep",
  19: "cow",
  20: "elephant",
  21: "bear",
  22: "zebra",
  23: "giraffe",
  24: "backpack",
  25: "umbrella",
  26: "handbag",
  27: "tie",
  28: "suitcase",
  29: "frisbee",
  30: "skis",
  31: "snowboard",
  32: "sports ball",
  33: "kite",
  34: "baseball bat",
  35: "baseball glove",
  36: "skateboard",
  37: "surfboard",
  38: "tennis racket",
  39: "bottle",
  40: "wine glass",
  41: "cup",
  42: "fork",
  43: "knife",
  44: "spoon",
  45: "bowl",
  46: "banana",
  47: "apple",
  48: "sandwich",
  49: "orange",
  50: "broccoli",
  51: "carrot",
  52: "hot dog",
  53: "pizza",
  54: "donut",
  55: "cake",
  56: "chair",
  57: "couch",
  58: "potted plant",
  59: "bed",
  60: "dining table",
  61: "toilet",
  62: "tv",
  63: "laptop",
  64: "mouse",
  65: "remote",
  66: "keyboard",
  67: "cell phone",
  68: "microwave",
  69: "oven",
  70: "toaster",
  71: "sink",
  72: "refrigerator",
  73: "book",
  74: "clock",
  75: "vase",
  76: "scissors",
  77: "teddy bear",
  78: "hair drier",
  79: "toothbrush"
}

COCO_COLOR_LIST = {
    0: (255, 56, 56),    1: (255, 157, 151),  2: (255, 112, 31),   3: (255, 178, 29),
    4: (207, 210, 49),   5: (72, 249, 10),    6: (146, 204, 23),   7: (61, 219, 134),
    8: (26, 147, 52),    9: (0, 212, 187),    10: (44, 153, 168),  11: (0, 194, 255),
    12: (52, 69, 147),   13: (100, 115, 255), 14: (0, 24, 236),    15: (132, 56, 255),
    16: (82, 0, 133),    17: (203, 56, 255),  18: (255, 149, 200), 19: (255, 55, 199),
    20: (255, 55, 199),  21: (255, 176, 59),  22: (255, 61, 112),  23: (255, 60, 60),
    24: (255, 179, 0),   25: (255, 222, 23),  26: (0, 255, 255),   27: (0, 255, 0),
    28: (255, 0, 255),   29: (255, 0, 0),     30: (0, 0, 255),     31: (0, 255, 0),
    32: (255, 255, 0),   33: (0, 255, 255),   34: (255, 0, 255),   35: (255, 128, 0),
    36: (128, 0, 255),   37: (0, 128, 255),   38: (128, 255, 0),   39: (0, 255, 128),
    40: (255, 0, 128),   41: (128, 0, 0),     42: (0, 128, 0),     43: (0, 0, 128),
    44: (128, 128, 0),   45: (128, 0, 128),   46: (0, 128, 128),   47: (64, 0, 0),
    48: (0, 64, 0),      49: (0, 0, 64),      50: (64, 64, 0),     51: (64, 0, 64),
    52: (0, 64, 64),     53: (192, 0, 0),     54: (0, 192, 0),     55: (0, 0, 192),
    56: (192, 192, 0),   57: (192, 0, 192),   58: (0, 192, 192),   59: (255, 64, 64),
    60: (64, 255, 64),   61: (64, 64, 255),   62: (255, 255, 64),  63: (255, 64, 255),
    64: (64, 255, 255),  65: (128, 128, 128), 66: (192, 128, 128), 67: (128, 192, 128),
    68: (128, 128, 192), 69: (192, 192, 128), 70: (192, 128, 192), 71: (128, 192, 192),
    72: (64, 128, 128),  73: (128, 64, 128),  74: (128, 128, 64),  75: (64, 128, 64),
    76: (64, 64, 128),   77: (192, 64, 64),   78: (64, 192, 64),   79: (64, 64, 192)
}

def radians_to_degrees(radians):
    """
    Converts radians to degrees.

    Args:
        radians (float): The angle in radians.

    Returns:
        float: The angle in degrees.
    """
    return radians * 180.0 / np.pi

def qlabel_to_cv_image(label):
    """
    Converts a QLabel image to a cv2 image.

    Args:
        label (QLabel): The QLabel containing the image.

    Returns:
        numpy.ndarray: The cv2 image.
    """
    if label is None:
        return None
    
    pixmap = label.pixmap()
    
    image = pixmap.toImage()

    return qimage_to_cv_image(image)
    
def qimage_to_cv_image(image:QImage):
    buffer = image.constBits()
    height = image.height()
    width = image.width()
    
    # Different QImage formats need different handling
    if image.format() == QImage.Format_RGB32:
        buffer.setsize(height * width * 4)
        arr = np.frombuffer(buffer, np.uint8).reshape((height, width, 4))
    elif image.format() == QImage.Format_ARGB32:
         buffer.setsize(height * width * 4)
         arr = np.frombuffer(buffer, np.uint8).reshape((height, width, 4))
    elif image.format() == QImage.Format_RGB888:
        buffer.setsize(height * width * 3)
        arr = np.frombuffer(buffer, np.uint8).reshape((height, width, 3))
    else:
        # Handle other formats if needed or raise an error
        raise ValueError(f"Unsupported image format:", image.format())

    # Convert to BGR for cv2 if necessary
    if arr.shape[2] == 4:
        cv_image = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    elif arr.shape[2] == 3:
        cv_image = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else: 
        cv_image = arr

    return cv_image

def cv_image_to_qimage(cv_image):
    """
    Converts a cv2 image to a QImage.

    Args:
        cv_image (numpy.ndarray): The cv2 image.

    Returns:
        QImage: The QImage.
    """
    if cv_image is None:
        return None
    if len(cv_image.shape) == 2:  # Grayscale image
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for consistency
    elif len(cv_image.shape) == 3 and cv_image.shape[2] == 4:  # RGBA image
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2BGR)
    elif len(cv_image.shape) == 3 and cv_image.shape[2] == 3:  # RGB image
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    elif len(cv_image.shape) == 3 and cv_image.shape[2] == 1:  # Single channel image
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for consistency
    else:
        raise ValueError(f"Unsupported cv2 image shape:", cv_image.shape)
    height, width, channel = cv_image.shape
    bytes_per_line = channel * width
    q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    
    return q_image

def cv_image_to_qlabel(cv_image):
    """
    Converts a cv2 image to a QLabel image.

    Args:
        cv_image (numpy.ndarray): The cv2 image.

    Returns:
        QImage: The QImage.
    """

    return cv_image_to_qimage(cv_image)


def string_to_cv_color_space(color_space):
    "converts a string to a cv2 color space assume the starting colorspace is RGB"

    #convert colorspace to all uppercase
    color_space = color_space.upper()

    if color_space == 'RGB':
        return cv2.COLOR_BGR2RGB
    elif color_space == 'GRAY':
        return cv2.COLOR_RGB2GRAY
    elif color_space == 'HSV':
        return cv2.COLOR_RGB2HSV
    elif color_space == 'YCRCB':
        return cv2.COLOR_RGB2YCrCb
    elif color_space == 'LAB':
        return cv2.COLOR_RGB2LAB
    elif color_space == 'YUV':
        return cv2.COLOR_RGB2YUV
    elif color_space == 'LUV':
        return cv2.COLOR_RGB2LUV
    else:
        raise ValueError(f"Unsupported color space:", color_space)

def cv_color_space_to_string(color_space):
    """
    Converts a cv2 color space to a string.

    Args:
        color_space (int): The cv2 color space.

    Returns:
        str: The string representation of the color space.
    """

    # print("color space:", color_space)
    # print("cv2.COLOR_BGR2RGB:", cv2.COLOR_BGR2RGB)
    if color_space == cv2.COLOR_BGR2RGB:
        return 'RGB'
    elif color_space == cv2.COLOR_BGR2GRAY:
        return 'GRAY'
    elif color_space == cv2.COLOR_BGR2HSV:
        return 'HSV'
    elif color_space == cv2.COLOR_BGR2YCrCb:
        return 'YCrCb'
    elif color_space == cv2.COLOR_BGR2LAB:
        return 'LAB'
    elif color_space ==cv2.COLOR_BGR2YUV:
        return 'YUV'
    elif color_space == cv2.COLOR_BGR2LUV:
        return 'LUV'
    else:
        raise ValueError(f"Unsupported color space:", color_space)


def to_rgb(image,cur_color_space):
    """
    Converts an image to RGB format.

    Args:
        image (numpy.ndarray): The image.

    Returns:
        numpy.ndarray: The RGB image.
    """

    if cur_color_space == 'RGB':
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif cur_color_space == 'BGR':
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif cur_color_space == 'GRAY':
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif cur_color_space == 'HSV':
        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    elif cur_color_space == 'YCrCb':
        return cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB)
    elif cur_color_space == 'LAB':
        return cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    elif cur_color_space == 'YUV':
        return cv2.cvtColor(image, cv2.COLOR_YUV2RGB)
    elif cur_color_space == 'LUV':
        return cv2.cvtColor(image, cv2.COLOR_LUV2RGB)
    else:
        raise ValueError(f"Unsupported color space:", cur_color_space)
    
def in_bounds(x, y, start_x,start_y,width, height):
    """
    Checks if a point is within the bounds of an image.

    Args:
        x (int): The x-coordinate.
        y (int): The y-coordinate.
        width (int): The width of the image.
        height (int): The height of the image.

    Returns:
        bool: True if the point is within the bounds of the image, False otherwise.
    """
    return start_x <= x < start_x + width and start_y <= y < start_y + height



def in_circle(x, y, center_x, center_y, radius):
    """
    Checks if a point is within a circle.

    Args:
        x (int): The x-coordinate.
        y (int): The y-coordinate.
        center_x (int): The x-coordinate of the circle's center.
        center_y (int): The y-coordinate of the circle's center.
        radius (int): The radius of the circle.

    Returns:
        bool: True if the point is within the circle, False otherwise.
    """
    return (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2


def clip_value(value, min_value, max_value):
    """
    Clips a value to a range.

    Args:
        value (int): The value to clip.
        min_value (int): The minimum value.
        max_value (int): The maximum value.

    Returns:
        int: The clipped value.
    """
    return max(min(value, max_value), min_value)


def qcolor_to_cv_color(qcolor,color_space='RGB'):
    """
    Converts a QColor to a cv2 color.

    Args:
        qcolor (QColor): The QColor.

    Returns:
        numpy.ndarray: The cv2 color.
    """
    # Get the RGB values from the QColor
    r, g, b, a = qcolor.getRgb()

    if color_space == 'RGB':
        # rgb_color = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_HSV2RGB)
        return np.array([[[r, g, b]]], dtype=np.uint8) 
    elif color_space == 'BGR':
        return np.array([b, g, r, a], dtype=np.uint8)
    elif color_space == 'HSV':
        hsv_color = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)
        return  hsv_color
    elif color_space == 'YCrCb':
        ycrcb_color = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2YCrCb)
        return ycrcb_color
    elif color_space == 'LAB':
        lab_color = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2LAB)
        return lab_color
    elif color_space == 'YUV':
        yuv_color = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2YUV)
        return yuv_color
    elif color_space == 'LUV':
        luv_color = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2LUV)
        return luv_color
    elif color_space == "RGBA":
        return np.array([[[r, g, b, a]]], dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported color space:", color_space)

def calculate_step_size(range):

    if range > 200:
        return 24
    
    if range > 180:
        return 12
    
    if range > 150:
        return 10
    
    if range > 100:
        return 8
    
    if range > 10:
        return 4
    
    if range < 10:
        return 2

def generate_subsample_rgb_colors(r_range=(0,256),g_range=(0,256),b_range=(0,256)):
    """Generates a subsample of RGB color tuples with a given step size."""
    top_range = int(abs(r_range[1] - r_range[0]))
    middle_range = int(abs(g_range[1] - g_range[0]))
    bottom_range = int(abs(b_range[1] - b_range[0]))

    # calculate step size of each channel to sample the colorspace without missing any colors and not overloading the computer    
    top_step = calculate_step_size(top_range)
    middle_step = calculate_step_size(middle_range)
    bottom_step = calculate_step_size(bottom_range)


    result = list(itertools.product(
        range(r_range[0], r_range[1],top_step), 
        range(g_range[0], g_range[1],middle_step), 
        range(b_range[0], b_range[1],bottom_step)))

    return result

def calculate_number_of_combinations(step=16,r_range=(0,256),g_range=(0,256),b_range=(0,256)):
    """Calculates the number of combinations of RGB colors with a given step size."""
    length_of_product = reduce(mul,(len(range(r_range[0], r_range[1], step)), len(range(g_range[0], g_range[1], step)), len(range(b_range[0], b_range[1], step))), 1)
    
    return length_of_product


def _normalize_range(range):
    min = range[0]
    max = range[1]

    if min == max:
        max = min + 1

    normalized_range = (min, max)
    
    return normalized_range

def get_colors_and_componets(color_space,c1_range = (0,256),c2_range = (0,256),c3_range = (0,256)):
        
        # normalize the ranges so that there is no zero range
        normalized_top_range = _normalize_range(c1_range)
        normalized_middle_range = _normalize_range(c2_range)
        normalized_bottom_range = _normalize_range(c3_range)

        permutations_array = np.array(generate_subsample_rgb_colors(r_range=normalized_top_range,g_range=normalized_middle_range,b_range=normalized_bottom_range))

        c1,c2,c3,colors = convert_colors_to_colorspace(permutations_array,color_space)

        return c1,c2,c3,colors
    
def convert_colors_to_colorspace(colors,color_space='RGB'):
        if color_space == 'RGB':
            return colors[:,0]/255.0,colors[:,1]/255.0,colors[:,2]/255.0,colors/255.0
        
        cv2_color_space = string_to_cv_color_space(color_space)
        # reshaping colors area for used by the cv2 function
        reshaped_colors = np.uint8(colors.reshape(-1,1,3))

        converted_colors = cv2.cvtColor(reshaped_colors, cv2_color_space)

        converted_colors = converted_colors.reshape(-1,3)/255.0

        c1 = converted_colors[:, 0]/255.0
        c2 = converted_colors[:, 1]/255.0
        c3 = converted_colors[:, 2]/255.0

        return c1,c2,c3,converted_colors



def calculate_angle_between(point1,point2):
    """
    Calculate the angle between two points with respect to the center point.

    Args:
        center (tuple): The center point (x, y).
        point1 (tuple): The first point (x, y).

    Returns:
        float: The angle in degrees.
    """
    delta_x = point2[0] - point1[0]
    delta_y = point2[1] - point1[1]

    angle = np.arctan2(delta_y, delta_x) * 180 / np.pi

    return angle


def calculate_point_along_arc(center, radius, angle):
    """
    Calculate the point along an arc given the center, radius, and angle.

    Args:
        center (tuple): The center point (x, y).
        radius (int): The radius of the arc.
        angle (float): The angle in degrees.

    Returns:
        tuple: The point along the arc (x, y).
    """
    angle_rad = np.deg2rad(angle)
    x = int(center[0] + radius * np.cos(angle_rad))
    y = int(center[1] + radius * np.sin(angle_rad))

    return x, y


def calculate_point_along_ellipse(center, radius_x, radius_y, angle):
    """
    Calculate a point along an elliptical curve given the center, radii, and angle.
    
    :param center: Tuple (x, y) representing the center of the ellipse
    :param radius_x: Horizontal radius of the ellipse
    :param radius_y: Vertical radius of the ellipse
    :param angle: Angle in degrees
    :return: Tuple (x, y) representing the point on the ellipse
    """
    # Convert the angle to radians
    angle_rad = np.radians(angle)
    
    # Calculate the x and y coordinates using the parametric equation of an ellipse
    x = int(center[0] + radius_x * np.cos(angle_rad))
    y = int(center[1] + radius_y * np.sin(angle_rad))  # Subtract because of inverted y-axis
    
    return x, y


# convert angle from -180 to 180 to 0 to 360
def convert_angle_to_360(angle):
    if angle < 0:
        angle = 360 + angle
    return angle

# given an angle based on a cloverwise direction, convert it to a counter clockwise direction
def convert_angle_to_counter_clockwise(angle):
    if angle > 180:
        angle = 360 - angle
    return angle


#flip the angle 180 degrees
def flip_angle(angle):
    if angle > 180:
        angle = 360 - angle
    else:
        angle = 360 + angle
    return angle




def non_max_suppression(boxes, overlapThresh):
    """
    Perform non-maximum suppression on a list of bounding boxes.

    Args:
        boxes (list): List of bounding boxes in the format (x, y, w, h).
        overlapThresh (float): Overlap threshold for suppression.

    Returns:
        list: List of bounding boxes after non-maximum suppression.
    """
    if len(boxes) == 0:
        return []

    # Convert boxes to numpy array
    for i, box_a in enumerate(boxes):
        area_a = box_a[2] * box_a[3]
        for j,box_b in enumerate(boxes,start=i+1):
            area_b = box_b[2] * box_b[3]
            # Calculate intersection area
            x1 = max(box_a[0], box_b[0])
            y1 = max(box_a[1], box_b[1])
            x2 = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
            y2 = min(box_a[1] + box_a[3], box_b[1] + box_b[3])
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            intersection = w * h

            union = area_a + area_b - intersection

            iou_score = intersection / union if union > 0 else 0
            # Suppress box_b if IoU is greater than the threshold
            if iou_score > overlapThresh:
                # Suppress box_b
                boxes[j] = (0, 0, 0, 0)
            # Calculate IoU
           
 