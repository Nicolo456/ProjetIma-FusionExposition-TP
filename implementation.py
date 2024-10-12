import numpy as np
from display_func import show_image, show_image_cv2, BGR2RGB, RGB2BGR, inspect_list_structure
from fs_func import open_image, save_image
from filters import apply_contrast_filter, apply_grayscale, apply_saturation_filter, apply_well_exposedness_filter
from weightmaps import calc_wm, get_wm, get_wms, normalize_wms, fuse_images, naive_fusion
from pyramid import pyramid_down, pyramid_up, laplacian_pyramid

# Mettre des s partout au pyr parce que y en a plusieurs
def main(imgs, show=False):
	wms = get_wms(imgs, show=False)
	wms = [[arr.astype(np.float32) for arr in sous_liste] for sous_liste in wms]
	
	imgs_pyrl  = [laplacian_pyramid(img) for img in imgs]
	"""print("===============imgs_pyrl===============")
	inspect_list_structure(imgs_pyrl)"""
	sorted_imgs_pyrl = sort_pyr(imgs_pyrl)
	"""print("===============sorted_imgs_pyrl===============")
	inspect_list_structure(sorted_imgs_pyrl)"""
	wms_pyrg = [pyramid_down(wm) for wm in wms]
	sorted_n_wms_pyr = get_sorted_n_wms_pyr(wms_pyrg)

	"""print("===============sorted_n_wms_pyr===============")
	inspect_list_structure(sorted_n_wms_pyr)"""
	sorted_fused_pyrs = get_sorted_fused_pyrs(sorted_n_wms_pyr, sorted_imgs_pyrl)

	print("===============sorted_fused_pyr===============")
	inspect_list_structure(sorted_fused_pyrs)
	fused_summed_pyr = get_fused_summed_pyr(sorted_fused_pyrs)

	if show:
		for i in range(len(fused_summed_pyr)):
			show_image(fused_summed_pyr[i], img1_title='Fused image')

	return fused_summed_pyr

def sort_pyr(pyrs):
	"""Sort pyramid by floor"""
	# Initialiser une liste vide pour stocker les groupes
	sorted_pyr = [[] for _ in range(len(pyrs[0]))]

	# Parcourir chaque sous-liste et regrouper les éléments par index
	for floor in pyrs:
		for i, img in enumerate(floor):
			sorted_pyr[i].append(img)
	return sorted_pyr


def get_sorted_n_wms_pyr(pyrs):                 
	sorted_pyr = sort_pyr(pyrs)
	sorted_n_wms_pyr = list(map(normalize_wms, sorted_pyr))
	return sorted_n_wms_pyr

def get_sorted_fused_pyrs(sorted_n_wms_pyr, sorted_imgs_pyrl):
	"""print("===============sorted_n_wms_pyr===============")
	inspect_list_structure(sorted_n_wms_pyr)
	print("===============sorted_imgs_pyrl===============")
	inspect_list_structure(sorted_imgs_pyrl)"""

	sorted_fused_pyrs = list(map(fuse_images, zip(sorted_imgs_pyrl), zip(sorted_n_wms_pyr))) # On applique fuse image sur chaque pyramide

	return sorted_fused_pyrs

def get_fused_summed_pyr(sorted_fused_pyr):
	"""On prend une liste de pyramide qu'on va mélanger avec une addition par étage
	@params: sorted_fused_pyr: [[image (np.array)]]"""

	fused_summed_pyr = []
	for pyr in sorted_fused_pyr:
		fused_summed_pyr.append(np.sum(pyr, axis=0))

	# fused_summed_pyr = list(map(lambda x: np.sum(x, axis=0), sorted_fused_pyr))
	return fused_summed_pyr

def get_final_image(imgs, show = False):
	fused_summed_pyr = main(imgs, show=show)
	print("===============fused_summed_pyr===============")
	inspect_list_structure(fused_summed_pyr)
	fused_summed_pyr_bgr = [RGB2BGR(img) for img in fused_summed_pyr]
	final_image = pyramid_up(fused_summed_pyr_bgr)
	final_image_rgb = BGR2RGB(final_image)
	if show:
		show_image(final_image, img1_title='Final image')
	return final_image

def old_pyramid_fusion(imgs, show = False):
	assert 0 == 1 # Pour eviter de l'utiliser sans faire exprès

	imgs_pyramids = [laplacian_pyramid(img) for img in imgs]

	imgs_pyramids_group = list(map(list, zip(*imgs_pyramids)))

	wms = get_wms(imgs, show=show)

	weight_pyramids = [pyramid_down(wm) for wm in wms]

	weights_pyramids_group = list(map(list, zip(*weight_pyramids)))

	n_weight_pyramids = [normalize_wms(wms, verbose=show) for wms in weight_pyramids]

	n_weight_pyramids_3d = [[np.stack(
		[n_wm] * 3, axis=-1) for n_wm in n_wms] 
		for n_wms in n_weight_pyramids]
	
	n_weight_pyramids_3d_group = list(map(list, zip(*n_weight_pyramids_3d)))

	fused_weight_pyramid = [np.sum(
		[n_wm_3d * img for n_wm_3d, img in zip(n_wms_3d, imgs)], axis=0) 
		for n_wms_3d, imgs in zip(n_weight_pyramids_3d_group, imgs_pyramids_group)]

	
	clipped_fused_weight_pyramids = [np.clip(
		fused_weight, 0, 255).astype(np.uint8) for fused_weight in fused_weight_pyramid]
	
	return clipped_fused_weight_pyramids

if __name__ == "__main__":
	 # Open an image with numpy, show it with matplotlib
	img_m = open_image("img/venise/MeanSat.jpg")
	img_o = open_image("img/venise/OverSat.jpg")
	img_u = open_image("img/venise/UnderSat.jpg")
	imgs = [img_m, img_o, img_u]

	fused_sum_pyr = main(imgs, show=True)
	#final_image = get_final_image(imgs, show=True)

	"""

	fused_image = naive_fusion(imgs, show=True)

	# Chargement image du papier
	img_p = open_image("img/venise/Result.jpg")

	# Afficher l'image fusionnée
	show_image(img_p, img1_title='Image issue du papier',
			   img2=fused_image, img2_title='Image issue de notre fusion naive')

	# Save the fused image
	save_image(fused_image, "img/venise/fused_image.jpg")

	# ============== Other example ===============
	imgs = []
	for i in range(4):
		imgs.append(open_image(f"img/chamber/iso{i + 1}.jpg"))

	fused_image = naive_fusion(imgs, show=False)

	# Chargement image du papier
	img_p = open_image("img/chamber/naive_paper_result.jpg")

	# Afficher l'image fusionnée
	show_image(img_p, img1_title='Image issue du papier',
			   img2=fused_image, img2_title='Image issue de notre fusion naive')

	# Save the fused image
	save_image(fused_image, "img/chamber/fused_image.jpg") 

	imgs = []
	for k in range(259, 266):
		imgs.append(open_image(f"img/perso_dams/DSC08{k}.tiff"))

	fused_image = naive_fusion(imgs, show=False)

	# Save the fused image
	save_image(fused_image, "img/perso_dams/fused_image.jpg")
	"""