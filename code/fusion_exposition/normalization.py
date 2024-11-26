# Simple decorator used to normalize the image in entry and exit of the function
import numpy as np


def normalise_decorator(func):
    """Decorator used to normalize the image in entry and return an image with 255 values"""
    def wrapper(img, *args, **kwargs):
        if img.dtype == np.uint8:
            img = img/255
            img = img.astype(np.float32)
        else:
            raise ValueError("The entry isn't a 255 int image")
        res = func(img, *args, **kwargs)

        if res.dtype == np.float32 and np.min(res) >= 0 and np.max(res) < 1:
            res = np.clip(res*255, 0, 255).astype(np.uint8)
        else:
            raise ValueError("The function must return a normalized float32 image. The output dtype is",
                             res.dtype, "Min:", np.min(res), "Max:", np.max(res), ".")
        return res
    return wrapper


def normalise_vector_decorator(force_normalize_return=False, quantile_loss=1):
    """Extend normalise_decorator to a list of images"""
    def decorator(func):
        def wrapper(imgs, *args, **kwargs):
            n_imgs = []
            for img in imgs:
                if img.dtype == np.uint8:
                    n_img = (img/255).astype(np.float32)
                    n_imgs.append(n_img)
                else:
                    raise ValueError("The entry isn't a 255 int images vector")
            res = func(n_imgs, *args, **kwargs)

            if force_normalize_return:
                min_with_loss = np.percentile(
                    res, quantile_loss)  # 10ème percentile
                max_with_loss = np.percentile(
                    res, 100 - quantile_loss)  # 90ème percentile
                # On oublie les valeurs trop basses et trop hautes
                n_res = ((res - min_with_loss) / (max_with_loss -
                         min_with_loss)).astype(np.float32)
                n_res = np.clip(n_res, 0, 1)
            else:
                n_res = res.astype(np.float32)

            if n_res.dtype == np.float32 and n_res.min() >= 0 and n_res.max() <= 1:
                n_res = np.clip(n_res*255, 0, 255).astype(np.uint8)
            else:
                raise ValueError("The function must return a normalized float32 image. The output dtype is",
                                 n_res.dtype, "Min:", np.min(n_res), "Max:", np.max(n_res), ".")
            return n_res
        return wrapper
    return decorator
