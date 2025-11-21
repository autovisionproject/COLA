import os
import shutil
import random
import argparse

def split_dataset(base_dir, split_ratio=0.8, seed=42):

    random.seed(seed)
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Iterate over class subfolders in base_dir
    for class_name in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_name)

        # Skip train/ and test/ folders
        if not os.path.isdir(class_path) or class_name in ["train", "test"]:
            continue

        images = os.listdir(class_path)
        random.shuffle(images)

        split_point = int(len(images) * split_ratio)
        train_images = images[:split_point]
        test_images = images[split_point:]

        # Create class subfolders inside train/ and test/
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        # Move images (instead of copying to save space)
        for img in train_images:
            shutil.move(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
        for img in test_images:
            shutil.move(os.path.join(class_path, img), os.path.join(test_dir, class_name, img))

        # Remove the now-empty original class folder
        os.rmdir(class_path)

    print("Dataset split completed inside:", base_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/test folders.")
    parser.add_argument("-base_dir", type=str, help="Path to dataset folder (e.g., imagenet-a)")
    parser.add_argument("-split_ratio", type=float, default=0.8, help="Ratio of training data (default=0.8)")
    parser.add_argument("-seed", type=int, default=42, help="Random seed for reproducibility (default=42)")
    
    args = parser.parse_args()

    split_dataset(args.base_dir, split_ratio=args.split_ratio, seed=args.seed)

