
import matplotlib.pyplot as plt
import numpy as np


# References
# Used below post to print images in my environment.
# https://stackoverflow.com/questions/56656777/userwarning-matplotlib-is-currently-using-agg-which-is-a-non-gui-backend-so

# Used below blog posts to understand np.mean() and np.where()
# https://www.sharpsightlabs.com/blog/numpy-mean/
# https://www.sharpsightlabs.com/blog/numpy-where/

# I got only images from below GitHub repository. I have never adapted any part of that code.
# You will see how complex that code when you check the repository.
# https://github.com/zftan0709/Hopfield-Network

# weight_calculation function returns weights for a single pattern.
# x_p should be an array.
def weight_calculation(x_p):
    # Assign N to pattern lengt.
    N = len(x_p)
    # Initialize list to store calculated weights
    w_ij = []
    for i in range(N):
        # Initialize w_temp list to store temporary weights.
        # All of them will be appended to list "w_ij".
        w_temp = []
        for j in range(N):
            # Calculate w_ij = (1/N) * x_i * x_j
            w = (1/N) * x_p[i] * x_p[j]
            w_temp.append(w)
        #print(w_temp)
        # Append calculated values to list "w_ij".
        w_ij.append(w_temp)
    return w_ij

# I created sgn() function for calculations.
# This will be used when calculation output: x_i
def sgn(x):
    if x >= 0:
        x = 1
    elif x < 0:
        x = -1
    return x

# Predict function x_j array, w_ij array as input.
# x_j is 1D image array. It will be given noisy images to predict the stored pattern.
def predict(x_j, w_ij):
    # Assign N to length of x_j. We will calculate all possible values.
    N = len(x_j)
    # Initialize list for "x_i" values.
    x_i = []
    for i in range(N):
        # Initialize x_temp for x_i calculations.
        x_temp = 0
        # Calculate x_i for all possible values and sum them.
        for j in range(N):
            x_temp += w_ij[i][j] * x_j[j]
        # Apply sgn() function. x_i = sgn(wij * x_j)
        x_temp = sgn(x_temp)
        # Append values to initialized list "x_i".
        x_i.append(x_temp)
    return np.array(x_i)

# This function is created for noisy images.
# It takes 1D array as input. And creates images according to noise type.
# If noise type = +1, you'll get normal image after predict() function.
# If noise type = -1, you'll get reversed image after predict() function.
def create_noisy_image(image_1D, noise_type):
    for i in range(len(image_1D)):
        if i > 256:
            image_1D[i] = noise_type
    return np.array(image_1D)

# Reads image and converts into 3D array
img = plt.imread(r"C:\Users\rkilimci\Documents\VSCodeFiles\hopfield-network\1.png")

# Takes mean of the 2nd axis of 3D array and converts it to 2D array shape(32, 32)
img = np.mean(img,axis=2)

# Takes mean of whole image. This generates one value.
img_mean = np.mean(img)

# Compares img values to img_mean.
# If lower than img_mean, it assigns -1.
# If greater than img_mean, it assigns +1.
img = np.where(img < img_mean, -1, 1)

# Show 2D array image which has +1s and -1s only.
# This makes image Yellow and Purple.
# This image will be given as STORED PATTERN to calculate weights.
plt.imshow(img)
plt.title("Stored pattern")
plt.show()

# Convert 2D array image to 1D for my calculation.
image_1D = img.flatten()

# Give 1D array image as input to weight_calculation() function.
weight_mat = weight_calculation(image_1D)

# Create noisy image for normal pattern prediction.
# You can check create_noisy_image() function for detailed explanation.
normal_noisy_img = create_noisy_image(image_1D, 1)
reversed_noisy_image = create_noisy_image(image_1D, -1)

# Convert noisy image to 2D array image
# Show noisy image to user
plt.imshow(normal_noisy_img.reshape((32,32)))
plt.title('Noisy Image')
plt.show()

# Give 1D array image as input to predict() function.
# Also give calculated weights weight_mat
x_i = predict(normal_noisy_img, weight_mat)
# print("x_i: ", x_i)

# Reshape 1D array output image to 2D array image
# Show predicted pattern to user
x_i_re = x_i.reshape((32,32))
plt.imshow(x_i_re)
plt.title('Predicted Pattern')
plt.show()

# Create noisy image for reversed pattern prediction.
# You can check create_noisy_image() function for detailed explanation.
reversed_noisy_image = create_noisy_image(image_1D, -1)

# Convert noisy image to 2D array image
# Show noisy image to user
plt.imshow(reversed_noisy_image.reshape((32,32)))
plt.title('Reversed Noisy Image')
plt.show()

# Give 1D array noisy image as input to predict() function.
# Also give calculated weights weight_mat
x_i = predict(reversed_noisy_image, weight_mat)

# Reshape 1D array output image to 2D array image
# Show predicted pattern to user
x_i_re = x_i.reshape((32,32))
plt.imshow(x_i_re)
plt.title('Reversed Predicted Pattern')
plt.show()