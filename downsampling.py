from PIL import Image
import os


def downsample_image(image_path, lambda_factor):
    # Ouvre l'image
    img = Image.open(image_path)

    # Récupère les dimensions originales
    width, height = img.size

    # Calcule les nouvelles dimensions
    new_width = int(width / lambda_factor)
    new_height = int(height / lambda_factor)

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


# Exemple d'utilisation
image_path = "img/perso_dams/DSC082"
lst = ["img/perso_dams/DSC082"+str(i)+".tiff" for i in range(59, 66)]
lambda_factor = 6  # Exemple : sous-échantillonner d'un facteur de 2
for img in lst:
    downsample_image(img, lambda_factor)
