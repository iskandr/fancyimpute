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
    KNN,
    MICE,
    BayesianRidgeRegression,
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


def rescale_pixel_values(images, order="C"):
    """
    Rescale the range of values in images to be between [0, 1]
    """
    images = np.asarray(images, order=order).astype("float32")
    images -= images.min()
    images /= images.max()
    return images


def color_balance(images):
    images = images.astype("float32")
    red = images[:, :, :, 0]
    green = images[:, :, :, 1]
    blue = images[:, :, :, 2]
    combined = (red + green + blue)
    total_color = combined.sum()
    overall_fraction_red = red.sum() / total_color
    overall_fraction_green = green.sum() / total_color
    overall_fraction_blue = blue.sum() / total_color

    for i in range(images.shape[0]):
        image = images[i]
        image_total = combined[i].sum()
        red_scale = overall_fraction_red / (red[i].sum() / image_total)
        green_scale = overall_fraction_green / (green[i].sum() / image_total)
        blue_scale = overall_fraction_blue / (blue[i].sum() / image_total)
        image[:, :, 0] *= red_scale
        image[:, :, 1] *= green_scale
        image[:, :, 2] *= blue_scale
    image[image < 0] = 0
    image[image > 255] = 255
    return images


class ResultsTable(object):

    def __init__(
            self,
            images_dict,
            percent_missing=0.25,
            saved_image_stride=25,
            dirname="face_images",
            scale_rows=False,
            center_rows=False):
        self.images_dict = images_dict
        self.labels = list(sorted(images_dict.keys()))
        self.images_array = np.array(
            [images_dict[k] for k in self.labels]).astype("float32")
        self.image_shape = self.images_array[0].shape
        self.width, self.height = self.image_shape[:2]
        self.color = (len(self.image_shape) == 3) and (self.image_shape[2] == 3)
        if self.color:
            self.images_array = color_balance(self.images_array)
        self.n_pixels = self.width * self.height
        self.n_features = self.n_pixels * (3 if self.color else 1)
        self.n_images = len(self.images_array)
        print("[ResultsTable] # images = %d, color=%s # features = %d, shape = %s" % (
            self.n_images, self.color, self.n_features, self.image_shape))

        self.flattened_array_shape = (self.n_images, self.n_features)

        self.flattened_images = self.images_array.reshape(self.flattened_array_shape)

        n_missing_pixels = int(self.n_pixels * percent_missing)

        missing_square_size = int(np.sqrt(n_missing_pixels))
        print("[ResultsTable] n_missing_pixels = %d, missing_square_size = %d" % (
            n_missing_pixels, missing_square_size))
        self.incomplete_images = remove_pixels(
            self.images_array,
            missing_square_size=missing_square_size)
        print("[ResultsTable] Incomplete images shape = %s" % (
            self.incomplete_images.shape,))
        self.flattened_incomplete_images = self.incomplete_images.reshape(
            self.flattened_array_shape)
        self.missing_mask = np.isnan(self.flattened_incomplete_images)
        self.normalizer = BiScaler(
            scale_rows=scale_rows,
            center_rows=center_rows,
            min_value=self.images_array.min(),
            max_value=self.images_array.max())
        self.incomplete_normalized = self.normalizer.fit_transform(
            self.flattened_incomplete_images)

        self.saved_image_indices = list(
            range(0, self.n_images, saved_image_stride))
        self.saved_images = defaultdict(dict)
        self.dirname = dirname
        self.mse_dict = {}
        self.mae_dict = {}

        self.save_images(self.images_array, "original", flattened=False)
        self.save_images(self.incomplete_images, "incomplete", flattened=False)

    def ensure_dir(self, dirname):
        if not exists(dirname):
            print("Creating directory: %s" % dirname)
            mkdir(dirname)

    def save_images(self, images, base_filename, flattened=True):
        self.ensure_dir(self.dirname)
        for i in self.saved_image_indices:
            label = self.labels[i].lower().replace(" ", "_")
            image = images[i, :].copy()
            if flattened:
                image = image.reshape(self.image_shape)
            image[np.isnan(image)] = 0
            figure = pylab.gcf()
            axes = pylab.gca()
            extra_kwargs = {}
            if self.color:
                extra_kwargs["cmap"] = "gray"
            assert image.min() >= 0, "Image can't contain negative numbers"
            if image.max() <= 1:
                image *= 256
            image[image > 255] = 255
            axes.imshow(image.astype("uint8"), **extra_kwargs)
            axes.get_xaxis().set_visible(False)
            axes.get_yaxis().set_visible(False)
            filename = base_filename + ".png"
            subdir = join(self.dirname, label)
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


def image_per_label(images, label_indices, label_names, max_size=2000):
    groups = defaultdict(list)
    for i, label_idx in enumerate(label_indices):
        label = label_names[label_idx].lower().strip().replace(" ", "_")
        groups[label].append(images[i])

    # as a pretty arbitrary heuristic, let's try taking the min variance
    # image for each person
    singe_images = {}
    for label, images in sorted(groups.items()):
        singe_images[label] = min(images, key=lambda image: image.std())
        if max_size and len(singe_images) >= max_size:
            break
    return singe_images


def get_lfw(max_size=None):
    dataset = fetch_lfw_people(color=True)
    # keep only one image per person
    return image_per_label(
        dataset.images,
        dataset.target,
        dataset.target_names,
        max_size=max_size)

if __name__ == "__main__":
    images_dict = get_lfw(max_size=2000)
    table = ResultsTable(
        images_dict=images_dict,
        scale_rows=False,
        center_rows=False)

    for negative_log_regularization_weight in [2, 3, 4]:
        regularization_weight = 10.0 ** -negative_log_regularization_weight
        table.add_entry(
            solver=MICE(
                n_nearest_columns=80,
                n_imputations=100,
                n_burn_in=5,
                model=BayesianRidgeRegression(lambda_reg=regularization_weight),
                init_fill_method="mean",
            ),
            name="MICE_%d" % negative_log_regularization_weight)

    for fill_method in ["mean", "median"]:
        table.add_entry(
            solver=SimpleFill(fill_method=fill_method),
            name="SimpleFill_%s" % fill_method)

    for k in [1, 3, 7]:
        table.add_entry(
            solver=KNN(
                k=k,
                orientation="rows"),
            name="KNN_k%d" % (k,))

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
