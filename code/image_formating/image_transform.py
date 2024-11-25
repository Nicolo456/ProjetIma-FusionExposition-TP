from cv2 import GaussianBlur, addWeighted


def unsharp_mask(image, sigma=1.0, strength=1.5, kernel_size=(5, 5)):
    # Appliquer un flou gaussien
    blurred = GaussianBlur(image, kernel_size, sigma)

    # Soustraire l'image floue de l'originale
    sharpened = addWeighted(image, 1.0 + strength, blurred, -strength, 0)

    return sharpened
