import time

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

from moviepy.editor import VideoFileClip
from IPython.display import HTML


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def find_fx(m, b, x):
    y = m*x + b
    return y


def find_fy(m, b, y):
    x = (y - b)/m
    return x


def draw_interpolated_line(img, x1_points, y1_points, x2_points, y2_points, color=[255, 0, 0], thickness=5):
    """
    This method will
    Args:
        img: Numpy array of the image
        x1_points: List of x1 points
        y1_points: List of y1_points
        x2_points: List of x2_points
        y2_points: List of y2_points
        color: Color of the line
        thickness: Thickness of the line
    """
    # Averaging points
    imshape = img.shape
    x1_mean = int(np.mean(x1_points))
    x2_mean = int(np.mean(x2_points))
    y1_mean = int(np.mean(y1_points))
    y2_mean = int(np.mean(y2_points))

    # Getting a line equation
    [m, b] = np.polyfit([x1_mean, x2_mean], [y2_mean, y1_mean], 1)

    # Now, based on the polynomial function found, we will extrapolate points to make the line longer
    x1_left = int(find_fy(m, b, imshape[0]))
    y1_left = int(imshape[0])
    x2_left = int(find_fy(m, b, imshape[1] * Constants.find_lane.max_y))
    y2_left = int(imshape[1] * Constants.find_lane.max_y)

    # Now drawing the image
    cv2.line(img, (x1_left, y1_left), (x2_left, y2_left), color, thickness)


def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    # Initialization
    x1l_points = []
    x2l_points = []
    y1l_points = []
    y2l_points = []
    x1r_points = []
    x2r_points = []
    y1r_points = []
    y2r_points = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            # Figuring out if the line is from the left or right lane
            a = (y2 - y1)/(x2 - x1)
            if np.isnan([x1, y1, x2, y2, a]).any():  # safety check
                continue
            if abs(a) < Constants.find_lane.min_a or abs(a) > Constants.find_lane.max_a:
                continue  # if the line is too inclined or not too much, probably not a lane, discard
            if a < 0:  # left lane points
                x1l_points.append(x1 if x1 < x2 else x2)
                x2l_points.append(x2 if x1 < x2 else x1)
                y1l_points.append(y1 if y1 < y2 else y2)
                y2l_points.append(y2 if y1 < y2 else y1)
            else:  # right lane points
                x1r_points.append(x1 if x1 < x2 else x2)
                x2r_points.append(x2 if x1 < x2 else x1)
                y1r_points.append(y1 if y1 > y2 else y2)
                y2r_points.append(y2 if y1 > y2 else y1)

    if x1l_points:  # treating left lane
        draw_interpolated_line(img, x1l_points, y1l_points, x2l_points, y2l_points, color, thickness)

    if x1r_points:  # treating right lane
        draw_interpolated_line(img, x1r_points, y1r_points, x2r_points, y2r_points, color, thickness)


def draw_polygon(img, vertices, color=[255, 0, 0], thickness=2):
    """
    Draws a polygon on a image
    Args:
        img (np.array): The image
        vertices (np.array): The vertices of the polygon
        color: Color of the polygon
        thickness: Thickness of the lines of the polygon
    """
    cv2.polylines(img, vertices, isClosed=True, color=color, thickness=thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, debug=False):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if lines is not None:
        draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.
def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def process_image(image, path=None):
    gs_img = grayscale(image)

    imshape = gs_img.shape
    gs_norm = gaussian_blur(gs_img, Constants.gaussian.kernel_size)
    canny_img = canny(gs_norm, Constants.canny.low_threshold, Constants.canny.high_threshold)
    vertices = np.array([[(imshape[1]*1/15,imshape[0]*14/15),(imshape[1]*7/15, imshape[0]*3/5), (imshape[1]*8/15, imshape[0]*3/5), (imshape[1]*14/15,imshape[0]*14/15)]], dtype=np.int32)
    masked_img = region_of_interest(canny_img, vertices)
    hough_image = hough_lines(masked_img, Constants.hough.rho, Constants.hough.theta, Constants.hough.threshold, Constants.hough.min_line_length, Constants.hough.max_line_gap)

    if path:
        masked_line = image.copy()
        draw_polygon(masked_line, vertices)
        plt.imsave(os.path.join("test_images_output", "_grayscale.".join(os.path.basename(path).split("."))), gs_img,
                   format="jpg", cmap="gray")
        plt.imsave(os.path.join("test_images_output", "_gaussian.".join(os.path.basename(path).split("."))), gs_norm,
                   format="jpg", cmap="gray")
        plt.imsave(os.path.join("test_images_output", "_canny.".join(os.path.basename(path).split("."))), canny_img,
                   format="jpg", cmap="gray")
        plt.imsave(os.path.join("test_images_output", "_masked.".join(os.path.basename(path).split("."))), masked_img,
                   format="jpg", cmap="gray")
        plt.imsave(os.path.join("test_images_output", "_masked_lines.".join(os.path.basename(path).split("."))),
                   masked_line, format="jpg")
        plt.imsave(os.path.join("test_images_output", "_hough.".join(os.path.basename(path).split("."))), hough_image,
                   format="jpg", cmap="gray")
    return weighted_img(hough_image, image)


class Constants:
    class gaussian:
        kernel_size = 11

    class canny:
        low_threshold = 15
        high_threshold = 45

    class hough:
        rho = 1
        theta = np.pi / 180
        threshold = 50
        min_line_length = 70
        max_line_gap = 180

    class find_lane:
        max_y = 1 / 3
        min_a = 0.3
        max_a = 4


if not os.path.isdir("test_images_output"):
    os.mkdir("test_images_output")

for image_file in os.listdir("test_images/"):
    image_path = os.path.join("test_images", image_file)
    image = mpimg.imread(image_path)
    result = process_image(image, image_path)
    plt.imsave(os.path.join("test_images_output", os.path.basename(image_path)), result, format="jpg")

for video in os.listdir("test_videos/"):
    video_path = os.path.join("test_videos", video)
    output = os.path.join("test_videos_output", video)
    clip = VideoFileClip(video_path)
    img = clip.get_frame(t=5)
    process_image(img, path=os.path.join("test_videos_output", "frame_" + os.path.basename(video_path).split(".")[0] + ".jpg"))
    yellow_clip = clip.fl_image(process_image)
    yellow_clip.write_videofile(output, audio=False)
