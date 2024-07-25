# Data Augmentation

Data augmentation is a technique used to artificially increase the size of a dataset by applying transformations to the original data. This technique is commonly used in computer vision tasks, where the transformations can be applied to images. The idea behind data augmentation is to increase the diversity of the dataset, which can help improve the performance of machine learning models.

In these tasks, we will explore different data augmentation techniques that can be applied to images. We will use the `imgaug` library, which provides a wide range of augmentation techniques that can be easily applied to images.

## TASKS

| Task                            | Description                                                                                             |
|---------------------------------|---------------------------------------------------------------------------------------------------------|
| [Flip image](./0-flip.py)       | Implement a function `flip_image(image)` that flips an image horizontally.                              |
| [Crop](./1-crop.py)             | Implement a function `crop_image(image, size)` that crops an image to a specified size.                 |
| [Rotate](./2-rotate.py)         | Implement a function `def rotate_image(image)` that rotates an image by a given angle of 90°.           |
| [Shear](./3-shear.py)           | Implement a function `def shear_image(image, intensity)` that shears an image by a given shear factor.  |
| [Brightness](./4-brightness.py) | Implement a function `def change_brightness(image, max_delta)` that adjusts the brightness of an image. |
| [Hue](./5-hue.py)               | Implement a function `def change_hue(image, delta)` that adjusts the hue of an image.                   |