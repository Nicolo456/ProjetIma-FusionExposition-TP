# HDR
HDR: a besoin de curves response de la caméra
HDR créé une nouvelle image avec une large dynamic range avant de faire un tone mapping alors que notre algorithme ne fait que fusionner nos images sur low dynamic range (lié au point précédent car pas nécessaire de calibrer la courbe de réponse)

HDR classique : Dans les méthodes HDR, la fusion des images est souvent guidée par des mesures physiques basées sur la lumière réelle capturée par la caméra, ce qui nécessite souvent des calculs complexes comme la récupération de la courbe de réponse de l'appareil photo.
Fusion d'exposition : Cette technique utilise des mesures subjectives de qualité d'image pour guider la fusion, telles que : 


# Pyramide de Laplace
Upsampling
Pas précisé dans articles mais bi-linéaire + courant et Burt et Adelson utilisent ça (bicubique peut aussi marcher). On applique un filtre gaussien après upsampling pour lisser les artefacts de l’interpolation.
General
Pyramide pour les poids et pour les images
Un carte de poids par image
Pyramide de Laplace utilise une pyramide Gaussienne puis upsampling

Processus:
Construction de la pyramide: a chaque étape on lisse avec un filtre Gaussien puis on sous échantillonne par deux l’image (on construit en fait une pyramide Gaussienne)
Remontée de la pyramide: on interpole l’image de résolution inférieur pour la réagrandir, on soustrait l’image agrandie à l’image d’origine et on stocke le résultat qui contient les détails de l’image → on a notre pyramide Laplacienne (interpole que d’un niveau)
Récupération de l’image: on interpole l’image de plus bas niveau, on y ajoute le détails du niveau supérieur, puis on interpole…

# Notes Nico:
On récupère pyramide des différences entre base et images upscaler de 1, qu’on multiplie avec pyramide des weight maps pour avoir fused pyramide
Fused pyramide -> final image
Et a la fin tu ajoutes chaque niveau de détail pour avoir image
