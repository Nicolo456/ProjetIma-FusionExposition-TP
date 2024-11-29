import matplotlib.pyplot as plt

"""
Ce module contient des fonctions permettant de:
- ouvrir une image
- sauvegarder une image"""

def open_image(filename):
    img = plt.imread(filename)
    return img


def save_image(img, filename):
    plt.imsave(filename, img)
