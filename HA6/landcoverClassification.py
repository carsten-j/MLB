# %% [markdown]
# You are supposed to train and apply a boosting model for landcover classification. Below, you can find some code that already parses the data. For your submission, run and submit the extended jupyter notebook.

# %%
import numpy as np

# %%
# load data
train = np.load("./train.npz")
X_train = train["X"]
y_train = train["y"].reshape((-1, 1))

test = np.load("./test.npz")
X_test = test["X"]
y_test = test["y"].reshape((-1, 1))

classes = {
    1: "cultivated_land",
    2: "forest",
    3: "grassland",
    4: "shrubland",
    5: "water",
    8: "artificial_surface",
    9: "bareland",
}


# %%
print("Shape of training data: {}".format(X_train.shape))
print("Shape of training labels: {}".format(y_train.shape))
print("Shape of test data: {}".format(X_test.shape))
print("Shape of test labels: {}".format(y_test.shape))

# %% [markdown]
# The training and test set contain a few thousand instances. Each instance is based on image of size 13x13 pixels, which are available for 12 timestamps and 6 bands. That is, one is given an array of shape (12,13,13,6) for each instance. The label arrays contain the labels associated with the instances, where the central pixel/position determines the class of the (whole) image array for each instance, see below.

# %%
# visualize some of the (image) data

import matplotlib.pyplot as plt

idx = 0

print(
    "Label (i.e., class of pixel in the center of the images) for image sequence: {}".format(
        classes[y_train[idx, 0]]
    )
)

for year in range(12):
    fig, axs = plt.subplots(1, 6, figsize=(24, 4))
    for b in range(6):
        axs[b].imshow(X_train[idx, year, :, :, b], cmap=plt.get_cmap("Greys"))
        axs[b].set_title("Band {}".format(b + 1))
