import cv2
import numpy as np

# Fonction pour aligner deux images


def align_images(img1, img2):
    # Convertir les images en niveaux de gris
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Utiliser SIFT pour détecter les points-clés et les descripteurs
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Utiliser BFMatcher pour trouver les correspondances entre les descripteurs
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Sélectionner les meilleurs correspondances
    good_matches = matches[:50]  # Ajustez selon les résultats

    # Extraire les points correspondants
    src_pts = np.float32(
        [keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Trouver l'homographie pour aligner img2 sur img1
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # Appliquer la transformation à img2
    height, width, channels = img1.shape
    aligned_img2 = cv2.warpPerspective(img2, M, (width, height))

    return aligned_img2

# Fonction pour superposer plusieurs images alignées


def stack_images(images):
    base_image = images[0]

    for img in images[1:]:
        aligned_img = align_images(base_image, img)
        base_image = cv2.addWeighted(base_image, 0.5, aligned_img, 0.5, 0)

    return base_image


if __name__ == "__main__":
    # Chargement des images
    # Remplacer par vos chemins d'images
    path_input = 'img/perso_dams/'
    path_output = 'img/trans_dams/'
    ext = '.tiff'
    img_names = {'med': "DSC08261_downsampled",
                 "over": "DSC08265_downsampled",
                 "under": "DSC08262_downsampled",
                 "med_over": "DSC08263_downsampled"}
    images = [cv2.imread(path_input+img_names[img_key]+ext)
              for img_key in img_names]

    base_image = images[0]
    aligned_imgs = [base_image]

    for img in images[1:]:
        aligned_imgs.append(align_images(base_image, img))

    for i, img_key in enumerate(img_names):
        # Sauvegarder ou afficher le résultat
        cv2.imwrite(path_output+img_key+"_aligned"+ext, aligned_imgs[i])

    for i, img_key in enumerate(img_names):
        cv2.imshow('Aligned and Stacked Image', aligned_imgs[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
