import os
import shutil
import argparse

def split_cub(base_path):
    # Metadata files
    images_file = os.path.join(base_path, "images.txt")
    labels_file = os.path.join(base_path, "image_class_labels.txt")
    split_file = os.path.join(base_path, "train_test_split.txt")
    classes_file = os.path.join(base_path, "classes.txt")

    # Output dirs
    train_dir = os.path.join(base_path, "train")
    test_dir = os.path.join(base_path, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Read images
    images = {}
    with open(images_file, "r") as f:
        for line in f:
            img_id, img_name = line.strip().split()
            images[int(img_id)] = img_name

    # Read labels
    labels = {}
    with open(labels_file, "r") as f:
        for line in f:
            img_id, class_id = line.strip().split()
            labels[int(img_id)] = int(class_id)

    # Read classes
    classes = {}
    with open(classes_file, "r") as f:
        for line in f:
            class_id, class_name = line.strip().split()
            classes[int(class_id)] = class_name

    # Read split
    splits = {}
    with open(split_file, "r") as f:
        for line in f:
            img_id, is_train = line.strip().split()
            splits[int(img_id)] = int(is_train)

    # Move images
    for img_id, img_name in images.items():
        class_id = labels[img_id]
        class_name = classes[class_id]

        src = os.path.join(base_path, "images", img_name)

        if splits[img_id] == 1:  # train
            dst_dir = os.path.join(train_dir, class_name)
        else:  # test
            dst_dir = os.path.join(test_dir, class_name)

        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(img_name))

        # Move instead of copy
        shutil.move(src, dst)

    print("All images moved into train/test/class_name folders!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split CUB_200_2011 dataset into train/test folders")
    parser.add_argument("-base_path", type=str, help="Path to CUB_200_2011 dataset")
    args = parser.parse_args()

    split_cub(args.base_path)

