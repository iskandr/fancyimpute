from os import mkdir
from os.path import exists, join
from collections import defaultdict

import pylab
from sklearn.datasets import fetch_lfw_people
import numpy as np

from fancyimpute import (
    SimpleFill,
    IterativeSVD,
    SoftImpute,
    BiScaler,
    DenseKNN,
)
from fancyimpute.common import masked_mae, masked_mse


def remove_pixels(
        full_images,
        missing_square_size=32,
        random_seed=0):
    np.random.seed(random_seed)
    incomplete_faces = []
    n_faces = len(full_images)
    height, width = full_images[0].shape[:2]
    for i in range(n_faces):
        image = full_images[i].copy()
        start_x = np.random.randint(
            low=0,
            high=height - missing_square_size + 1)
        start_y = np.random.randint(
            low=0,
            high=width - missing_square_size + 1)
        image[
            start_x: start_x + missing_square_size,
            start_y: start_y + missing_square_size] = np.nan
        incomplete_faces.append(image)
    return np.array(incomplete_faces, dtype=np.float32)


class ResultsTable(object):

    def __init__(
            self,
            images,
            percent_missing=0.25,
            saved_image_stride=125,
            dirname="face_images",
            scale_rows=False,
            center_rows=False):
        self.images = np.asarray(images, order="C").astype("float32")
        self.image_shape = images[0].shape
        self.width, self.height = self.image_shape[:2]
        self.color = (len(self.image_shape) == 3) and (self.image_shape[2] == 3)
        self.n_pixels = self.width * self.height
        self.n_features = self.n_pixels * (3 if self.color else 1)
        self.n_images = len(self.images)
        print("[ResultsTable] # images = %d, color=%s # features = %d, shape = %s" % (
            self.n_images, self.color, self.n_features, self.image_shape))

        self.flattened_array_shape = (self.n_images, self.n_features)

        self.flattened_images = self.images.reshape(self.flattened_array_shape)

        n_missing_pixels = int(self.n_pixels * percent_missing)

        missing_square_size = int(np.sqrt(n_missing_pixels))
        print("[ResultsTable] n_missing_pixels = %d, missing_square_size = %d" % (
            n_missing_pixels, missing_square_size))
        self.incomplete_images = remove_pixels(
            self.images,
            missing_square_size=missing_square_size)
        print("[ResultsTable] Incomplete images shape = %s" % (
            self.incomplete_images.shape,))
        self.flattened_incomplete_images = self.incomplete_images.reshape(
            self.flattened_array_shape)
        self.missing_mask = np.isnan(self.flattened_incomplete_images)
        self.normalizer = BiScaler(
            scale_rows=scale_rows,
            center_rows=center_rows,
            min_value=self.images.min(),
            max_value=self.images.max())
        self.incomplete_normalized = self.normalizer.fit_transform(
            self.flattened_incomplete_images)

        self.saved_image_indices = list(
            range(0, self.n_images, saved_image_stride))
        self.saved_images = defaultdict(dict)
        self.dirname = dirname
        self.mse_dict = {}
        self.mae_dict = {}

        self.save_images(self.images, "original", flattened=False)
        self.save_images(self.incomplete_images, "incomplete", flattened=False)

    def rescale_pixel_values(self, images, order="C"):
        """
        Rescale the range of values in images to be between [0, 1]
        """
        images = np.asarray(images, order=order).astype("float32")
        images -= images.min()
        images /= images.max()
        return images

    def ensure_dir(self, dirname):
        if not exists(dirname):
            print("Creating directory: %s" % dirname)
            mkdir(dirname)

    def save_images(self, images, base_filename, flattened=True):
        self.ensure_dir(self.dirname)
        for i in self.saved_image_indices:
            image = images[i, :].copy()
            if flattened:
                image = image.reshape(self.image_shape)
            image[np.isnan(image)] = 0
            figure = pylab.gcf()
            axes = pylab.gca()
            extra_kwargs = {}
            if self.color:
                extra_kwargs["cmap"] = "gray"
            axes.imshow((image * 256).astype("uint8"), **extra_kwargs)
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
            X_true=self.flattened_images,
            X_pred=completed,
            mask=self.missing_mask)
        mse = masked_mse(
            X_true=self.flattened_images,
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


def unique_images(images, labels, max_size=1000):
    result = []
    seen_labels = set([])
    for i, label in enumerate(labels):
        if label in seen_labels:
            continue
        result.append(images[i])
        seen_labels.add(label)
        if max_size and len(seen_labels) >= max_size:
            break
    return np.array(result)


def get_lfw(n=None):
    dataset = fetch_lfw_people(color=True)
    # keep only one image per person
    return unique_images(dataset.images, dataset.target, max_size=n)


if __name__ == "__main__":
    images = get_lfw(n=1000)
    table = ResultsTable(images)

    for k in [1, 5, 9]:
        table.add_entry(
            solver=DenseKNN(
                k=k,
                orientation="rows"),
            name="DenseKNN_k%d" % (k,))

    for fill_method in ["mean", "median"]:
        table.add_entry(
            solver=SimpleFill(fill_method=fill_method),
            name="SimpleFill_%s" % fill_method)

    for shrinkage_value in [25, 50, 100]:
        # SoftImpute without rank constraints
        table.add_entry(
            solver=SoftImpute(
                shrinkage_value=shrinkage_value),
            name="SoftImpute_lambda%d" % (shrinkage_value,))

    for rank in [10, 20, 40]:
        table.add_entry(
            solver=IterativeSVD(
                rank=rank,
                init_fill_method="zero"),
            name="IterativeSVD_rank%d" % (rank,))

    table.save_html_table()
    table.print_sorted_errors()
