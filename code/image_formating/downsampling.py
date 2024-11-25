from PIL import Image
import os
import numpy as np


def downsample_image(image_path, lambda_factor, two_divisibily_factor=0):
    """Downsample an image by a given factor.
    Args:
        image_path (str): Chemin vers l'image à sous-échantillonner.
        lambda_factor (int): Facteur de sous-échantillonnage.
        two_divisibily_factor (int): Facteur de divisibilité par 2. Oblige l'image à être divisible par 2^two_divisibily_factor.
    """
    # Ouvre l'image
    img = Image.open(image_path)

    # Récupère les dimensions originales
    width, height = img.size

    # Calcule les nouvelles dimensions
    new_width = (int(width / lambda_factor)//2 **
                 two_divisibily_factor) * 2**two_divisibily_factor
    new_height = (int(height / lambda_factor)//2 **
                  two_divisibily_factor) * 2**two_divisibily_factor

    # Redimensionne (sous-échantillonne) l'image
    downsampled_img = img.resize(
        (new_width, new_height), Image.Resampling.LANCZOS)

    # Sépare le nom de fichier et son extension
    base, ext = os.path.splitext(image_path)

    # Crée le nouveau nom de fichier avec "_downsampled"
    new_image_path = f"{base}_downsampled{ext}"

    # Enregistre l'image sous-échantillonnée
    downsampled_img.save(new_image_path)

    print(f"Image enregistrée sous: {new_image_path}")


def upsample_image(image_path, two_divisibily_factor=0):
    """Upsample an image by a given factor.
    Args:
        image_path (str): Chemin vers l'image à sous-échantillonner.
        two_divisibily_factor (int): Facteur de divisibilité par 2. Oblige l'image à être divisible par 2^two_divisibily_factor.
    """
    # Ouvre l'image
    img = Image.open(image_path)

    # Récupère les dimensions originales
    width, height = img.size

    # Trouve le multiple de 2**two_divisibily_factor le plus proche de la largeur et de la hauteur
    new_width = (int(width / 2 ** two_divisibily_factor)
                 * 2 ** two_divisibily_factor)
    new_height = (int(height / 2 ** two_divisibily_factor)
                  * 2 ** two_divisibily_factor)

    # Redimensionne (sous-échantillonne) l'image
    upsampled_img = img.resize(
        (new_width, new_height), Image.Resampling.LANCZOS)

    upsampled_np_img = np.array(upsampled_img)
    return upsampled_np_img


if __name__ == "__main__":

    # Exemple d'utilisation
    image_path = "img/perso_dams/DSC082"
    lst = ["img/perso_dams/DSC082"+str(i)+".tiff" for i in range(59, 66)]
    # Facteur de sous-échantillonnage
    lambda_factor = 2
    # Facteur obligant l'image à être divisible par 2**two_divisibily_factor
    two_divisibily_factor = 10
    for img in lst:
        downsample_image(img, lambda_factor,
                         two_divisibily_factor=two_divisibily_factor)
