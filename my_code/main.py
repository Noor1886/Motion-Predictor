from pathlib import Path
from linear_regression import linear_regression
from neural_algo import neural_network
import numpy as np
import pandas as pd
import os


class Model:
    @classmethod
    def preprocess(cls, dataset):
        dataset = np.array(dataset)
        sample = np.empty([len(dataset), 21], dtype=object)
        print(f"Dataset Length: {len(dataset)}")
        ranges = [
            (0, 108),
            (108, 216),
            (216, 324),
            (324, 432),
            (432, 540),
            (540, 648),
            (648, 756),
            (756, 864),
            (864, 972),
            (972, 1080),
        ]
        for i in range(len(dataset)):
            for j in range(10):
                sum_c = 0
                avg_c = 0
                range_c = ranges[j]
                for k in range(range_c[0], range_c[1]):
                    sum_c += dataset[i][k]
                    avg_c = sum_c / 108
                sample[i][j] = avg_c
            sample[i][10] = dataset[i][1080]
            sample[i][11] = dataset[i][1081]
            sample[i][12] = dataset[i][1082] + dataset[i][1083]
            sample[i][13] = dataset[i][1084]
            sample[i][14] = dataset[i][1085]
            sample[i][15] = dataset[i][1086] + dataset[i][1087]
            sample[i][16] = dataset[i][1088]
            sample[i][17] = dataset[i][1089]
            sample[i][18] = dataset[i][1090] + dataset[i][1091]
            sample[i][19] = dataset[i][1092]
            sample[i][20] = dataset[i][1093]
        sample = pd.DataFrame(sample)
        return sample

    @classmethod
    def build_test_data(cls):
        path = os.path.join(os.getcwd(), "Final Data", "testing_combine")
        csv_files = os.listdir(path)
        print(f"CSV Files: {csv_files}")

        output_file = os.path.join(os.getcwd(), "Final Data", "Test.csv")
        if os.path.exists(output_file) is False:
            Path(output_file).touch()

        cls.__generate_csv(path, csv_files, output_file)

    @classmethod
    def generate_samples(cls):
        cls.build_test_data()
        cls.build_train_data()

    @classmethod
    def build_train_data(cls):
        # The corridor_CSV folder is expected to be located at `Final Data/Training Data/corridor_CSV/`
        corridor_path = os.path.join(os.getcwd(), "Final Data", "combine_corridor")
        open_box_path = os.path.join(os.getcwd(), "Final Data", "Open_Box_CSV")
        special_CSV = os.path.join(os.getcwd(), "Final Data", "special_CSV")
        csv_files = os.listdir(corridor_path)
        print(f"CSV Files: {csv_files}")

        output_file = os.path.join(os.getcwd(), "Final Data", "Train_Temp.csv")
        if os.path.exists(output_file) is False:
            Path(output_file).touch()

        temp = 0
        for file in csv_files:
            if Path(file).suffix == ".csv":
                csv_data = pd.read_csv(os.path.join(corridor_path, file))
                rows_csv_data, columns_csv_data = csv_data.shape
                temp += rows_csv_data
                ex_data = cls.preprocess(csv_data)
                ex_data.to_csv(output_file, mode="a", header=None, index=False)
        csv_files = os.listdir(open_box_path)
        temp = 0
        for file in csv_files:
            if Path(file).suffix == ".csv":
                csv_data = pd.read_csv(os.path.join(open_box_path, file))
                rows_csv_data, columns_csv_data = csv_data.shape
                temp += rows_csv_data
                ex_data = cls.preprocess(csv_data)
                ex_data.to_csv(output_file, mode="a", header=None, index=False)
        
        csv_files = os.listdir(special_CSV)
        temp = 0
        for file in csv_files:
            if Path(file).suffix == ".csv":
                csv_data = pd.read_csv(os.path.join(special_CSV, file))
                rows_csv_data, columns_csv_data = csv_data.shape
                temp += rows_csv_data
                ex_data = cls.preprocess(csv_data)
                ex_data.to_csv(output_file, mode="a", header=None, index=False)

    @classmethod
    def linear_regression(cls):
        path_to_train_temp_csv, path_to_test_csv = cls.__get_train_and_test_paths()

        # Run the linear regression predictor
        linear_regression(path_to_train_temp_csv, path_to_test_csv)

    @classmethod
    def neural_network(cls):
        path_to_train_temp_csv, path_to_test_csv = cls.__get_train_and_test_paths()

        # Run the neural network predictor
        neural_network(path_to_train_temp_csv, path_to_test_csv)

    @classmethod
    def __generate_csv(cls, base_path, csv_files, output_file):
        temp = 0
        for file in csv_files:
            if Path(file).suffix == ".csv":
                csv_data = pd.read_csv(os.path.join(base_path, file))
                rows_csv_data, columns_csv_data = csv_data.shape
                temp += rows_csv_data
                ex_data = cls.preprocess(csv_data)
                ex_data.to_csv(output_file, mode="a", header=None, index=False)

    @classmethod
    def __get_train_and_test_paths(cls):
        path_to_test_csv = os.path.join(os.getcwd(), "Final Data", "Test.csv")
        if os.path.exists(path_to_test_csv) is False:
            raise f"File Test.csv not found in {path_to_test_csv}"

        path_to_train_temp_csv = os.path.join(os.getcwd(), "Final Data", "Train_Temp.csv")
        if os.path.exists(path_to_train_temp_csv) is False:
            raise f"File Train_Temp.csv not found in {path_to_train_temp_csv}"

        return path_to_train_temp_csv, path_to_test_csv


if __name__ == "__main__":
    # Generate the training data and test data combined
    Model.generate_samples()

    # Run Linear Regression Predictor
    Model.linear_regression()

    # Run Neural Network Predictor
    Model.neural_network()
