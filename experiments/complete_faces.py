from os import mkdir
from os.path import exists, join

import pylab
from sklearn.datasets import fetch_olivetti_faces
import numpy as np

from fancyimpute import (
    AutoEncoder,
    MICE,
    MatrixFactorization,
    NuclearNormMinimization,
    SimpleFill,
    IterativeSVD,
    SoftImpute,
)


def load_faces_data(
        missing_square_size=32,
        width=64,
        height=64,
        fetch_data_fn=fetch_olivetti_faces,
        random_seed=0):
    np.random.seed(random_seed)
    dataset = fetch_data_fn()
    full_faces_matrix = dataset.data.astype(np.float32)
    incomplete_faces = []
    n_faces, n_pixels = full_faces_matrix.shape
    assert n_pixels == width * height
    for i in range(n_faces):
        image = full_faces_matrix[i].reshape((height, width)).copy()
        start_x = np.random.randint(low=0, high=height - missing_square_size + 1)
        start_y = np.random.randint(low=0, high=width - missing_square_size + 1)
        image[
            start_x:start_x + missing_square_size,
            start_y:start_y + missing_square_size] = np.nan
        incomplete_faces.append(image.reshape((n_pixels,)))
    incomplete_faces_matrix = np.array(incomplete_faces, dtype=np.float32)
    return full_faces_matrix, incomplete_faces_matrix


def save_images(
        images,
        base_filename,
        imshape=(64, 64),
        image_indices=[0, 50, 100, 150, 200, 250, 300, 350],
        dirname="face_images"):
    if not exists(dirname):
        mkdir(dirname)

    paths = []
    for i in image_indices:
        fig = pylab.gcf()
        ax = pylab.gca()
        image = images[i, :].copy().reshape(imshape)
        image[np.isnan(image)] = 0
        ax.imshow(image, cmap="gray")
        filename = base_filename + "_%d" % (i) + ".png"
        subdir = join(dirname, str(i))
        if not exists(subdir):
            mkdir(subdir)
        path = join(subdir, filename)
        fig.savefig(path)
        paths.append(path)
    return paths


if __name__ == "__main__":
    original, incomplete = load_faces_data()
    save_images(original, base_filename="original")
    save_images(incomplete, base_filename="incomplete")

    for fill_method in ["mean", "median"]:
        filler = SimpleFill(fill_method=fill_method)
        completed_fill = filler.complete(incomplete)
        save_images(
            completed_fill,
            base_filename="SimpleFill_%s" % fill_method)

    for fill_method in ["zero"]:
        for shrinkage_value in [10, 20, 40, 80]:
            print("Fill=%s, shrinkage=%d" % (fill_method, shrinkage_value))
            # SoftImpute without rank constraints
            save_images(
                SoftImpute(
                    init_fill_method=fill_method,
                    shrinkage_value=shrinkage_value,
                    min_value=0,
                    max_value=1).complete(incomplete),
                base_filename="SoftImpute_%s_lambda%d" % (
                    fill_method, shrinkage_value))

    for rank in [5, 50]:
        for fill_method in ["zero"]:
            save_images(
                IterativeSVD(
                    rank=rank,
                    init_fill_method=fill_method,
                    min_value=0,
                    max_value=1,
                ).complete(incomplete),
                base_filename="IterativeSVD_%s_rank%d" % (
                    fill_method,
                    rank))
        for l1_fraction in [10, 100]:
            for l2_fraction in [10, 100]:
                save_images(
                    MatrixFactorization(
                        rank,
                        l1_penalty=1.0 / l1_fraction,
                        l2_penalty=1.0 / l2_fraction,
                        min_value=0,
                        max_value=1).complete(incomplete),
                    base_filename="MatrixFactorization_rank%d_l1_%d_l2_%d" % (
                        rank,
                        l1_fraction,
                        l2_fraction))

    for rank in [5, 50]:
        save_images(
            AutoEncoder(
                hidden_layer_sizes=[100, rank],
                hidden_activation="tanh",
                output_activation="sigmoid",
                min_value=0,
                max_value=1,
            ).complete(incomplete),
            base_filename="nn_rank%d" % rank)
