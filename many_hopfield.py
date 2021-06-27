import matplotlib.pyplot as plt
import numpy as np

# I defined image paths first.
number_1_path = r"C:\Users\rkilimci\Documents\VSCodeFiles\hopfield-network\1.png"
number_2_path = r"C:\Users\rkilimci\Documents\VSCodeFiles\hopfield-network\2.png"
number_3_path = r"C:\Users\rkilimci\Documents\VSCodeFiles\hopfield-network\3.png"

# conver_image() function takes image and converts it into 2D array image.
def convert_image(image_path):
    # Reads image and converts into 3D array
    img = plt.imread(image_path)

    # Takes mean of the 2nd axis of 3D array and converts it to 2D array shape(32, 32)
    img = np.mean(img,axis=2)

    # Takes mean of whole image. This generates one value.
    img_mean = np.mean(img)

    # Compares img values to img_mean.
    # If lower than img_mean, it assigns -1.
    # If greater than img_mean, it assigns +1.
    img = np.where(img < img_mean, -1, 1)

    return img

# Convert number "1" image to 2D array image
number_1_converted = convert_image(number_1_path)
# Show 2D array image which has +1s and -1s only.
# This makes image Yellow and Purple.
# This image will be given as ONE OF THE STORED PATTERNS to calculate weights.
plt.imshow(number_1_converted)
plt.title("Stored pattern number 1")
plt.show()

# Convert number "2" image to 2D array image
number_2_converted = convert_image(number_2_path)
# Show 2D array image which has +1s and -1s only.
# This makes image Yellow and Purple.
# This image will be given as ONE OF THE STORED PATTERNS  to calculate weights.
plt.imshow(number_2_converted)
plt.title("Stored pattern number 2")
plt.show()

# Convert number "3" image to 2D array image
number_3_converted = convert_image(number_3_path)
# Show 2D array image which has +1s and -1s only.
# This makes image Yellow and Purple.
# This image will be given as ONE OF THE STORED PATTERNS  to calculate weights.
plt.imshow(number_3_converted)
plt.title("Stored pattern number 3")
plt.show()

# Convert 2D array images to 1D images for my calculation.
image_1D_number_1 = number_1_converted.flatten()
image_1D_number_2 = number_2_converted.flatten()
image_1D_number_3 = number_3_converted.flatten()

# Concatenate all 1D image arrays into matrix
# all_pattern array shape = (3, 1024)
all_pattern = np.vstack((image_1D_number_1, image_1D_number_2, image_1D_number_3))
print(all_pattern.shape)
print(len(all_pattern[0]))

# Calculate weights for all patterns.
def weight_calculation(x_p):
    # Assign N one of the image lentgh. N = 1024
    N = len(x_p[0])
    # Assign pattern = image count. pattern = 3 in our case
    pattern = len(x_p)
    # Initialize list to store calculated weights
    w_ij = []
    for i in range(N):
        w_temp = []
        for j in range(N):
            w_sum = 0
            # Calculate weights for each pattern
            for z in  range(pattern):
                w_sum += x_p[z][i] + x_p[z][j]
            w = w_sum / N
            w_temp.append(w)
        # Append calculated weights into one list. w_ij.shape = (1024, 1024)
        w_ij.append(w_temp)
    return np.array(w_ij)

# Calculate weight using our defined weight_calculation() function.
weight_mat = weight_calculation(all_pattern)
#print(weight_mat.shape)


# I created sgn() function for calculations.
# This will be used when calculating output: x_i
def sgn(x):
    if x >= 0:
        x = 1
    elif x < 0:
        x = -1
    return x

# Predict function: x_j array = image will be predicted, w_ij array = weight matrix
# x_j is 1D image array. It will be given normal or noisy images to predict the stored pattern.
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

###################################
###### TEST WITH NORMAL IMAGE ######
###################################

# Convert noisy image to 2D array image
# Show noisy image to user
plt.imshow(image_1D_number_1.reshape((32,32)))
plt.title('Image will be predicted: Number 1(Without noise)')
plt.show()

# Give 1D array image as input to predict() function.
# Also give calculated weights weight_mat
x_i = predict(image_1D_number_1, weight_mat)
# print("x_i: ", x_i)

# Reshape 1D array output image to 2D array image
# Show predicted pattern to user
x_i_re = x_i.reshape((32,32))
plt.imshow(x_i_re)
plt.title('Predicted Pattern: Number 1(Without noise)')
plt.show()

######################################
###### END OF NORMAL IMAGE TEST ######
######################################

######################################

###################################
###### TEST WITH NOISY IMAGE ######
###################################
# Create noisy image for normal pattern prediction.
# You can check create_noisy_image() function for detailed explanation.
normal_noisy_img = create_noisy_image(image_1D_number_1, -1)

# Convert noisy image to 2D array image
# Show noisy image to user
plt.imshow(normal_noisy_img.reshape((32,32)))
plt.title('Image will be predicted: Number 1(With noise)')
plt.show()

# Give 1D array image as input to predict() function.
# Also give calculated weights weight_mat
x_i = predict(normal_noisy_img, weight_mat)
# print("x_i: ", x_i)

# Reshape 1D array output image to 2D array image
# Show predicted pattern to user
x_i_re = x_i.reshape((32,32))
plt.imshow(x_i_re)
plt.title('Predicted Pattern Number 1(With noise)')
plt.show()

#####################################
###### END OF NOISY IMAGE TEST ######
#####################################