import pickle
import numpy as np
import matplotlib.pyplot as plt

LOAD_DIR = "./training_data"


def read_pickle(directory, filename, finger_index, validate=False):
    long_file_name = f"{directory}/{filename}_{finger_index}{'_validate' if validate else ''}.pkl"

    with open(long_file_name, "rb") as file:
        output = pickle.load(file)
        print(f"Loaded {type(output)} from {long_file_name}")

    return output


def main():
    finger_index = 730
    sdf_points = read_pickle(LOAD_DIR, "sdf_points", finger_index, False)
    sdf_values = read_pickle(LOAD_DIR, "sdf_values", finger_index, False)

    sdf_points_0 = sdf_points[0]
    sdf_values_0 = sdf_values[0]

    plt.figure()
    plt.grid()
    plt.title("SDF values histogram after filtering for first frame")
    plt.xlabel("sdf values")
    plt.ylabel("Counts")
    plt.hist(sdf_values_0)
    plt.plot([0, 0], [0, 100000], label="0 line")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
