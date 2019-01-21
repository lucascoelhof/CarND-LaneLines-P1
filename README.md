# **Finding Lane Lines on the Road** 

---

During the beginning of the self-driving car Engineer Nanodegree, I have learning a series of image processing techniques.
We had the opportunity to apply this knowledge in a very interesting and hands-on project, to find traffic lanes on
images and in a video stream. For that, I have programmed a image processing pipeline to treat the images
and video streams to detect the lanes.


[initial]: ./test_images/solidYellowLeft.jpg "Initial image"
[grayscale]: ./test_images_output/solidYellowLeft_grayscale.jpg "Grayscale"
[gaussian]: ./test_images_output/solidYellowCurve_gaussian.jpg "Gaussian filter"
[canny]: ./test_images_output/solidYellowLeft_canny.jpg "Canny"
[mask]: ./test_images_output/solidYellowCurve_masked.jpg "Mask"
[masked_lines]: ./test_images_output/solidYellowLeft_masked_lines.jpg "Final result"
[final]: ./test_images_output/solidYellowLeft.jpg "Final result"

---

## Image processing pipeline

My pipeline consisted of 5 steps. We start with this image, and on each step I will
show what did the step did with the image.

![alt text][initial]


### 1. Converting to grayscale

Converting to grayscale allows us to handle the image more easily with the techniques that will follow. 
I've used `cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)`, the simplest way to convert to grayscale. In the future,
I might re-visit this part of the code and use more sophisticated transformations or maybe a method based
in clustering and thresholding, like [Otsu's method](https://en.wikipedia.org/wiki/Otsu%27s_method).

![alt text][grayscale]


### 2. Gaussian filtering
In this step, I applied a gaussian filter, which smoothes the image and creates a blur.
This is necessary to reduce image noise and remove unwanted results.
I've used `cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)`,
with `kernel_size=11`. A larger kernel would apply a stronger smoothing.

![alt text][gaussian]


### 3. Canny detection
The [Canny edge detector](https://en.wikipedia.org/wiki/Canny_edge_detector) is a method to find edges in a image.
It is based on the gradient of pixel intensities. I've used `cv2.Canny(img, low_threshold, high_threshold)`.
`low_threshold` and `high_threshold` are parameters set to filter the gradient values, so only values between
`low_threshold` and `high_threshold` are kept. This part was critical, because depending on the road conditions,
the Canny algorithm would ignore the lanes, so a proper calibration is essential.

![alt text][canny]


### 4. Mask

On this project, we assume that the camera is mounted in a fixed and know position of the car. Given that,
we can expect that the lanes will be inside a polygon:

![alt text][masked_lines]

Given that, we can filter the canny transform result to have the following result.
![alt text][mask]
We can see that much of the ambient, like trees and the horizon, which would trick our algorithm, is removed from
the image.

### 5. Hough transform and final result

The [Hough transform](https://en.wikipedia.org/wiki/Hough_transform) is used for feature extraction in computer vision to find classes of shapes in a given
picture. These shapes can be circles, polygons, but in this case, I used Hough transform to detect lines from the masked
Canny result image to find the actual driving lanes. After using the Hough transform, we will also filter the lines,
to detect the right and left lane lines, using a average filter and a line interpolation. The final result is:

![alt text][final]


## Try it yourself

Please follow instructions [here](https://github.com/udacity/CarND-Term1-Starter-Kit) to set up your __conda__ environment.
Then, on your terminal use `conda activate carnd-term1` to activate the conda environment, then run `jupyter notebook`
to see the results in jupyter or just run the P1.py file using `python3 P1.py`. The images will appear on __test_images_output__
folder and the videos in __test_videos_output__.


## Shortcomings of the pipeline
This pipeline was specifically designed and calibrated for this specific set of example images and videos.
Thus, in different lighting conditions, or even different pavement color, this algorithm can behave unexpectedly.
I personally don't expect that it will work at all in low light conditions like in the dark.
Moreover, objects on the road or even dirt can arguably confuse the pipeline.

## Possible improvements
One immediate improvement I can propose is to find a way to automatically calibrate the canny transform
based on the current lighting of the image. I could see that there are some pages about auto-canny, like 
[here](https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/)
but I expect that during this Self-Driving Car Engineer Nanodegree we will learn more exciting and effective 
techniques to treat that. Another possible improvement would be to have a configuration file instead of the `Configuration`
class I'm using. I haven't done that because I did not want to include other dependencies than the ones from the 
course example, but in a field implementation I would definitely use configuration files instead of
changing the code for that.

