# DO NOT RENAME THIS FILE
# This file enables automated judging
# This file should stay named as `submission.py`
#
# TAMU Datathon Challenge: Puzzle Solver
#
# Author: Cihat Kececi
#

# Import Python Libraries
import os
import numpy as np
from glob import glob
from PIL import Image
from itertools import permutations

# Import helper functions from utils.py
import utils


class Predictor:
    """
    DO NOT RENAME THIS CLASS
    This class enables automated judging
    This class should stay named as `Predictor`
    """

    def __init__(self):
        """
        Initializes any variables to be used when making predictions
        """
        self.all_permutations = list(permutations(range(0, 4)))

    def make_prediction(self, img_path):
        """
        DO NOT RENAME THIS FUNCTION
        This function enables automated judging
        This function should stay named as `make_prediction(self, img_path)`

        INPUT:
            img_path:
                A string representing the path to an RGB image with dimensions 128x128
                example: `example_images/1.png`

        OUTPUT:
            A 4-character string representing how to re-arrange the input image to solve the puzzle
            example: `3120`
        """

        # Load the image
        img = Image.open(img_path)

        img_array = np.asarray(img, dtype=np.float32)

        split_img = utils.get_uniform_rectangular_split(img_array, 2, 2)

        def vec_dist(x, y):
            """
            This function is used for comparing the edges of the four pieces
            """
            std = np.mean(np.std(x, axis=(0, 1)) + np.std(y, axis=(0, 1)))

            if std < 4:
                return np.inf

            return np.mean(np.abs(x - y)) / std

        def distance(top_left, top_right, bottom_left, bottom_right):
            """"
            Calculate the sum of the distances for a given permutation.
            """
            return vec_dist(top_left[:, -1, :], top_right[:, 0, :]) \
                   + vec_dist(top_left[-1, :, :], bottom_left[0, :, :]) \
                   + vec_dist(top_right[-1, :, :], bottom_right[0, :, :]) \
                   + vec_dist(bottom_left[:, -1, :], bottom_right[:, 0, :])

        prediction = np.zeros(24)

        # Calculate the distances for each permutation
        for i, perm in enumerate(self.all_permutations):
            src = np.argsort(perm)
            prediction[i] = distance(*[split_img[j] for j in src])

        # Select the permutation with the least distance
        result = self.all_permutations[np.argmin(prediction)]
        return ''.join(str(x) for x in result)


def main():
    # It assumes that the train dataset is located at `./train`

    # Local imports since they are optional dependencies
    import multiprocessing as mp

    PARALLEL_RUN = True

    true_count, false_count = 0, 0

    predictor = Predictor()

    folders = os.listdir('train')
    for i, folder_name in enumerate(folders):
        print(f'Evaluating folder {i + 1}/{len(folders)}')

        if PARALLEL_RUN:
            pool = mp.Pool()
            preds = pool.map(predictor.make_prediction, glob(f'train/{folder_name}/*'))
            for pred in preds:
                if pred == folder_name:
                    true_count += 1
                else:
                    false_count += 1

        else:
            for img_name in glob(f'train/{folder_name}/*'):
                prediction = predictor.make_prediction(img_name)

                # print(f'{prediction} <=> {folder_name}')
                if prediction == folder_name:
                    true_count += 1
                else:
                    false_count += 1

        print(f'True : {true_count}')
        print(f'False: {false_count}')
        print(f'Acc: {true_count / (true_count + false_count):.4f}')


if __name__ == '__main__':
    main()
