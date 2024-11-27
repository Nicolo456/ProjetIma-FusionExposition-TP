import os
from PIL import Image
from downsampling import downsample_image

if __name__ == "__main__":
    # Chemin du dossier contenant les images .tiff
    DOSSIER_SOURCE = "hidden_imgs/compressed"
    # Chemin du dossier où les images .jpg seront sauvegardées
    DOSSIER_DEST = "hidden_imgs/compressed"

    # Crée le dossier de destination s'il n'existe pas
    os.makedirs(DOSSIER_DEST, exist_ok=True)

    # Parcourt tous les fichiers dans le dossier source
    for nom_fichier in os.listdir(DOSSIER_SOURCE):
        # Vérifie si le fichier a l'extension .tiff ou .tif
        if nom_fichier.lower().endswith(('.tiff', '.tif')):
            chemin_fichier = os.path.join(DOSSIER_SOURCE, nom_fichier)
            # Ouvre l'image
            img = downsample_image(chemin_fichier, 1, save=False)
            # Définit le nouveau nom pour l'image convertie
            nouveau_nom = os.path.splitext(nom_fichier)[0] + ".jpg"
            chemin_nouveau_fichier = os.path.join(
                DOSSIER_DEST, nouveau_nom)
            # Convertit et sauvegarde l'image au format JPG
            img.convert("RGB").save(chemin_nouveau_fichier, "JPEG")
            print(f"Converti : {nom_fichier} -> {nouveau_nom}")

    print("Conversion terminée !")
