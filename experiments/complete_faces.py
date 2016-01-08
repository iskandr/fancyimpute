from os import mkdir
from os.path import exists, join
from collections import defaultdict

import pylab
from sklearn.datasets import fetch_olivetti_faces
import numpy as np

from fancyimpute import (
    # AutoEncoder,
    MICE,
    # MatrixFactorization,
    # NuclearNormMinimization,
    SimpleFill,
    IterativeSVD,
    SoftImpute,
    BiScaler,
    DenseKNN,
)
from fancyimpute.common import masked_mae, masked_mse


def remove_pixels(
        full_images,
        width=64,
        height=64,
        missing_square_size=32,
        random_seed=0):
    np.random.seed(random_seed)
    incomplete_faces = []
    n_faces, n_pixels = full_images.shape
    assert n_pixels == width * height
    for i in range(n_faces):
        image = full_images[i].reshape((height, width)).copy()
        start_x = np.random.randint(
            low=0,
            high=height - missing_square_size + 1)
        start_y = np.random.randint(
            low=0,
            high=width - missing_square_size + 1)
        image[
            start_x:start_x + missing_square_size,
            start_y:start_y + missing_square_size] = np.nan
        incomplete_faces.append(image.reshape((n_pixels,)))
    incomplete_faces_matrix = np.array(incomplete_faces, dtype=np.float32)
    return incomplete_faces_matrix


class ResultsTable(object):

    def __init__(
            self,
            images,
            width=64,
            height=64,
            missing_square_size=25,
            saved_image_stride=75,
            dirname="face_images"):
        self.original = self.rescale_pixel_values(images)

        self.width = width
        self.height = height
        assert images.shape[1] == width * height
        self.incomplete = remove_pixels(
            images,
            width=width,
            height=height,
            missing_square_size=missing_square_size)
        self.missing_mask = np.isnan(self.incomplete)
        self.normalizer = BiScaler(scale_rows=False, min_value=0, max_value=1)
        self.incomplete_normalized = self.normalizer.fit_transform(
            self.incomplete)

        self.n_images = len(original)
        self.saved_image_indices = list(
            range(0, self.n_images, saved_image_stride))
        self.saved_images = defaultdict(dict)
        self.dirname = dirname
        self.mse_dict = {}
        self.mae_dict = {}

        self.save_images(self.original, "original")
        self.save_images(self.incomplete, "incomplete")

    def rescale_pixel_values(self, images):
        """
        Rescale the range of values in images to be between [0, 1]
        """
        images = np.asarray(images).copy()
        images = images.astype("float32")
        images -= images.min()
        images /= images.max()
        return images

    def ensure_dir(self, dirname):
        if not exists(dirname):
            print("Creating directory: %s" % dirname)
            mkdir(dirname)

    def save_images(self, images, base_filename):
        imshape = (self.width, self.height)
        self.ensure_dir(self.dirname)
        for i in self.saved_image_indices:
            image = images[i, :].copy().reshape(imshape)
            image[np.isnan(image)] = 0
            figure = pylab.gcf()
            axes = pylab.gca()
            axes.imshow(image, vmin=0, vmax=1, cmap="gray")
            axes.get_xaxis().set_visible(False)
            axes.get_yaxis().set_visible(False)
            filename = base_filename + "_%d" % (i) + ".png"
            subdir = join(self.dirname, str(i))
            self.ensure_dir(subdir)
            path = join(subdir, filename)
            figure.savefig(
                path,
                bbox_inches='tight')
            self.saved_images[i][base_filename] = path

    def add_entry(self, solver, name):
        print("Running %s" % name)
        completed_normalized = solver.complete(self.incomplete_normalized)
        completed = self.normalizer.inverse_transform(completed_normalized)

        mae = masked_mae(
            X_true=self.original,
            X_pred=completed,
            mask=self.missing_mask)
        mse = masked_mse(
            X_true=self.original,
            X_pred=completed,
            mask=self.missing_mask)
        print("==> %s: MSE=%0.4f MAE=%0.4f" % (name, mse, mae))
        self.mse_dict[name] = mse
        self.mae_dict[name] = mae
        self.save_images(completed, base_filename=name)

    def sorted_errors(self):
        """
        Generator for (rank, name, MSE, MAE) sorted by increasing MAE
        """
        for i, (name, mae) in enumerate(
                sorted(self.mae_dict.items(), key=lambda x: x[1])):
            yield(i + 1, name, self.mse_dict[name], self.mae_dict[name],)

    def print_sorted_errors(self):
        for (rank, name, mse, mae) in self.sorted_errors():
            print("%d) %s: MSE=%0.4f MAE=%0.4f" % (
                rank,
                name,
                mse,
                mae))

    def save_html_table(self, filename="results_table.html"):
        html = """
            <table>
            <th>
                <td>Rank</td>
                <td>Name</td>
                <td>Mean Squared Error</td>
                <td>Mean Absolute Error</td>
            </th>
        """
        for (rank, name, mse, mae) in self.sorted_errors():
            html += """
            <tr>
                <td>%d</td>
                <td>%s</td>
                <td>%0.4f</td>
                <td>%0.4f</td>
            </tr>
            """ % (rank, name, mse, mae)
        html += "</table>"
        self.ensure_dir(self.dirname)
        path = join(self.dirname, filename)
        with open(path, "w") as f:
            f.write(html)
        return html

if __name__ == "__main__":
    dataset = fetch_olivetti_faces()
    original = dataset.data

    table = ResultsTable(original)

    for fill_method in ["mean", "median"]:
        table.add_entry(
            solver=SimpleFill(fill_method=fill_method),
            name="SimpleFill_%s" % fill_method)

    table.add_entry(
        solver=MICE(
            n_imputations=100,
            approximate_but_fast_mode=True),
        name="MICE")

    for k in [1, 3, 5]:
        table.add_entry(
            solver=DenseKNN(
                k=k,
                orientation="rows"),
            name="DenseKNN_k%d" % (k,))

    for shrinkage_value in [25, 50, 100]:
        # SoftImpute without rank constraints
        table.add_entry(
            solver=SoftImpute(
                shrinkage_value=shrinkage_value),
            name="SoftImpute_lambda%d" % (shrinkage_value,))

    for rank in [5, 50]:
        table.add_entry(
            solver=IterativeSVD(
                rank=rank,
                init_fill_method="zero"),
            name="IterativeSVD_rank%d" % (rank,))
        """
        table.add_entry(
            solver=AutoEncoder(
                hidden_layer_sizes=[200, rank],
                hidden_activation="tanh",
                output_activation="linear",
                patience_epochs=25,
                n_burn_in_epochs=0,
                recurrent_weight=1.0,
                missing_input_noise_weight=0,
            ),
            name="AutoEncoder_rank%d" % (rank,))
        table.add_entry(
            solver=MatrixFactorization(
                rank,
                l1_penalty=0.1,
                l2_penalty=0.1),
            name="MatrixFactorization_rank%d" % rank)
        """

    table.save_html_table()
    table.print_sorted_errors()
