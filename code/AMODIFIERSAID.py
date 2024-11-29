from main import init_img_venise, init_img_dam, init_img_foyer, get_compressed_image, view_exposition_modified_image
from func.arg_parser import init_arg_parser
from fusion_exposition.display_func import show_image

if __name__ == "__main__":
    args = init_arg_parser()  # Initialisation des arguments (ne pas modifier)

    # ------------------------- Commenter le code pour prendre la photo voulu ---------------------
    #imgs, img_m = init_img_venise()
    #imgs, img_m = init_img_dam()
    imgs, img_m = init_img_foyer()
    # ----------------------------------------------------------------------------------------

    # ------------------------------------- MODIFIER ICI -------------------------------------------
    """Description des arguments à modifier:
    imgs: listes des images utilisé dans la fusion
    power_coef: liste des coefficients de puissance pour les différents paramètres
        ↳ dans cette ordre [contrast_power, saturation_power, well_exposedness_power]
    is_naive_fusion: booléen, si True, utilise la fusion naïve, sinon utilise la fusion avec les poids
     """
    result = get_compressed_image(
        imgs, [1, 1, 1], args, is_naive_fusion=True)  # Ne pas modifier args
    show_image(result, img1_title='Resultat final')  # Montre l'image
    # ----------------------------------------------------------------------------------------

    # ------------------------------------- MODIFIER ICI -------------------------------------------
    """Description des arguments à modifier:
    sharped: booléen, si True, utilise un filtre de sharpening pour réduire le flou (utile pour les photos "dams")
    save: booléen, si True, sauvegarde les images avec chaque état
     """
    view_exposition_modified_image(result, imgs, args, sharped=True, save=False)
    # ATTENTION: pour voir les différents filtres colorimétriques, lancer la commande avec le flag -e (`python3 AMODIFIERSAID.py -e`). Pour éviter de recalculer les images vous pouver ajouter le flag -r (`python3 AMODIFIERSAID.py -e -r`).
    