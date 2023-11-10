import os
import shutil
from sklearn.model_selection import train_test_split


def copy_files(file_list, original_dir, new_dir):
    for file in file_list:
        original_file_path = os.path.join(original_dir, file)
        shutil.copy(original_file_path, new_dir)


def split_data(all_txt_dir, all_img_dir):
    # define directories
    train_txt_dir = os.path.join(all_txt_dir, "train")
    train_img_dir = os.path.join(all_img_dir, "train")
    test_txt_dir = os.path.join(all_txt_dir, "test")
    test_img_dir = os.path.join(all_img_dir, "test")

    # make directories
    os.makedirs(train_txt_dir, exist_ok=True)
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(test_txt_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)

    all_txt_file = [
        txt_file for txt_file in os.listdir(all_txt_dir) if txt_file.endswith(".txt")
    ]

    # split data into train and test
    train_txt_files, test_txt_files = train_test_split(
        all_txt_file, test_size=0.3, random_state=42
    )
    print(len(train_txt_files), len(test_txt_files))

    # create train
    copy_files(train_txt_files, all_txt_dir, train_txt_dir)
    train_img_files = [txt_file.replace(".txt", ".jpg") for txt_file in train_txt_files]
    copy_files(train_img_files, all_img_dir, train_img_dir)

    # create test
    copy_files(test_txt_files, all_txt_dir, test_txt_dir)
    test_img_files = [txt_file.replace(".txt", ".jpg") for txt_file in test_txt_files]
    copy_files(test_img_files, all_img_dir, test_img_dir)


if __name__ == "__main__":
    pass
