**Vehicle Detection Project**

Note: My implementation of project 5 uses a deep learning approach, instead of using the HOG based approach. https://discussions.udacity.com/t/deep-learning-approach-to-p5/245483/5

### Model Architecture
My models architecture was based off the U-net design, with progressively deeper convolutions, before combining them with earlier layers. This allowed the model to identify high level features, while still being able to localise the position of them within the picture.

The model takes an image of size 600x960 and returns an array of the same size, with each pixel being the likelyhood of their being a car in that pixel.

I fed the input frame through the model (after downsizing to a 600x960 image), and receive a 600x960 array that corresponds to a per-pixel detection of the vehicles in the car.

This "whole image input" allows us to only have to feed each frame into the model once, rather then having to do a sliding window over the image and rerunning the model. This also means we don't have to merge multiple windows.

### Training

The model was trained on Udacity's two self driving car datasets found on https://github.com/udacity/self-driving-car/tree/master/annotations.

The model was trained for 15 epochs using a generator.

The dataset consists of a series of images, with each image having accompanying bounding boxes for each car or vehicle in the image.

We convert these rectangles into a 960x600x1 mask, like so.

[orig]: ./output_images/original_image.png
![alt text][orig]

[mask]: ./output_images/mask.png
![alt text][mask]

The aim of the model is to predict this mask for any given image.

### Result

Anecdotally, the model would always see all nearby cars, but would also have a moderate tendency to make false positive guesses. Luckily these could be filtered out later by filtering out boxes of a small size. This is discussed further in a later section.

The following example from the test set demonstrates the problem.

Input Image

[test_original]: ./output_images/test_original.png
![alt text][test_original]

Correct mask
[test_actual]: ./output_images/test_actual.png
![alt text][test_actual]

Model Mask
[test_guess]: ./output_images/test_guess.png
![alt text][test_guess]


### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](https://youtu.be/snDnKzVgbMc)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I fed the input frame through the model (after downsizing to a 600x960 image), and receive a 600x960 array that corresponds to a per-pixel detection of the vehicles in the car.

I then average this against the last 10 received heatmaps, and apply a threshold.

This average is then fed through scipy's label function (`scipy.ndimage.measurements.label()`)

These labels are converted to bounding boxes, and any box with an area of under 300 pixels is excluded as extraneous.

Here is an example frame from the video. The blurred edges of the image indicate pixels where different frames disagree on if a car exists at that location.

[image5]: ./output_images/bounding_boxes.png
![alt text][image5]

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I spent a great deal of time working on the HOG method of dealing with identifying the cars in the image, but had a difficult time handling tweaking hyper parameters due to low performance making it difficult to quickly iterate.

Implementing U-net improved performance by taking then outputed frames per second from roughly 0.5 to 4. This can likely attributed to the test machine having a relatively slow CPU (a 7 year old Xeon X5650) with low single thread performance, whilst having a relatively overpowered GPU (Nvidia 1080 Ti). As the HOG calculations were all being done on the CPU, and the NN was being run on the GPU, using the NN played to the strengths of the computational power on hand.

The model has plenty of room for improvement. As it only returns a single "car or no car" response, calculating labels results in overlapping cars being seen as a single car. This could be improved on by either having the model guess the number of cars at a single pixel, or by using a method that calculates some sort of "ratio of rectangular bounding box detected as car" so that multiple bounding boxes can be pulled out of a single label.

Another area for improvement is augmentation. While the dataset contains 25k images, their is room for more augmentation to cover things like weather, shadows, or tweaking scale.
