import numpy as np

"""
Ce module contient différents décorateurs qui permettent de vérifier les points suivants:
- vérifier qu'une/des images/pyramides sont bien normalisées
- vérifier que la taille d'une image est bien divisible par 2^(d'étages)
- executer la fonction avec la valeur maximale de nombre d'étages
- vérifier que l'image est en niveau de gris """


def assert_normalized_image(nb_image=1, negative=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            images = args[:nb_image]
            for image in images:
                assert isinstance(
                    image, np.ndarray) and image.dtype == np.float32, "Input must be a NumPy array of float32."
                if negative:
                    assert np.all((image >= -1) & (image <= 1)
                                  ), f"Image must be normalized with values in [-1, 1]. here image have values in ({np.min(image)},{np.max(image)})"
                else:
                    assert np.all((image >= 0) & (
                        image <= 1)), f"Image must be normalized with values in [0, 1]. here image have values in ({np.min(image)},{np.max(image)})"
            return func(*args, **kwargs)
        return wrapper
    return decorator


def assert_normalized_images(nb_images=1, negative=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            imagess = args[:nb_images]
            for images in imagess:
                for image in images:
                    assert isinstance(
                        image, np.ndarray) and image.dtype == np.float32, "Input must be a NumPy array of float32."
                    if negative:
                        assert np.all(
                            (image >= -1) & (image <= 1)), f"Image must be normalized with values in [-1, 1]. here image have values in ({np.min(image)},{np.max(image)})"
                    else:
                        assert np.all((image >= 0) & (
                            image <= 1)), f"Image must be normalized with values in [0, 1]. here image have values in ({np.min(image)},{np.max(image)})"
            return func(*args, **kwargs)
        return wrapper
    return decorator


def assert_normalized_pyr(nb_pyr=1, negative=False):
    """A pyramid is a list of images, each image is a 'floor' of the pyramid"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            pyrs = args[:nb_pyr]
            for pyr in pyrs:
                assert isinstance(
                    pyr, list), "Inputs must be a list of NumPy arrays."
                for floor_img in pyr:
                    assert isinstance(
                        floor_img, np.ndarray) and floor_img.dtype == np.float32, "Input must be a NumPy array of float32."
                    if negative:
                        assert np.all((floor_img >= -1) & (floor_img <= 1)
                                      ), f"Image must be normalized with values in [-1, 1]. here image have values in ({np.min(floor_img)},{np.max(floor_img)})"
                    else:
                        assert np.all((floor_img >= 0) & (
                            floor_img <= 1)), f"Image must be normalized with values in [0, 1]. here image have values in ({np.min(floor_img)},{np.max(floor_img)})"
            return func(*args, **kwargs)
        return wrapper
    return decorator


def assert_normalized_pyrs(nb_pyrs=1, negative=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            pyrss = args[:nb_pyrs]
            for pyrs in pyrss:
                assert isinstance(
                    pyrs, list), "Inputs must be a list of NumPy arrays."
                for pyr in pyrs:
                    assert isinstance(
                        pyr, list), "Inputs must be a list of NumPy arrays."
                    for floor_img in pyr:
                        assert isinstance(
                            floor_img, np.ndarray) and floor_img.dtype == np.float32, "Input must be a NumPy array of float32."
                        if negative:
                            assert np.all((floor_img >= -1) & (floor_img <= 1)
                                          ), f"Image must be normalized with values in [-1, 1]. here image have values in ({np.min(floor_img)},{np.max(floor_img)})"
                        else:
                            assert np.all((floor_img >= 0) & (
                                floor_img <= 1)), f"Image must be normalized with values in [0, 1]. here image have values in ({np.min(floor_img)},{np.max(floor_img)})"
            return func(*args, **kwargs)
        return wrapper
    return decorator


def assert_image_size_divisible(func):
    def wrapper(imgs, floors, *args, **kwargs):
        i = floors - 1
        error_msg = f"La taille de l'image doit être divisible par 2^(floors-1) \n\t=> ici {
            2**i} ne divise pas {imgs[0].shape[0:2]}"
        assert imgs[0].shape[0] % 2**i == 0 and imgs[0].shape[1] % 2**i == 0, error_msg

        return func(imgs, floors, *args, **kwargs)
    return wrapper


def get_max_floor_if_floor_Max(func):
    def wrapper(imgs, floors, *args, **kwargs):
        if floors == "Max":
            i = 0
            while imgs[0].shape[0] % 2**i == 0 and imgs[0].shape[1] % 2**i == 0:
                i += 1
            n_floors = i - 1
            print(
                f"[INFO] Le nombre de niveaux de la pyramide est : {n_floors}")
        return func(imgs, n_floors, *args, **kwargs)
    return wrapper


def is_img_greyscale(img):
    """Check if an image is greyscale or not
    @param: img: [np.array] an image
    @return: [bool] True if the image is greyscale, False otherwise"""
    return not (len(img.shape) > 2 and img.shape[2] == 3)
