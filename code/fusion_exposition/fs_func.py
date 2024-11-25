import matplotlib.pyplot as plt


def open_image(filename):
    img = plt.imread(filename)
    return img


def save_image(img, filename):
    plt.imsave(filename, img)
