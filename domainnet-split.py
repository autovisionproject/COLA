import os
import shutil
import argparse

# Parse base_path from command line
parser = argparse.ArgumentParser(description="Reorganize DomainNet dataset into train/test/class folders")
parser.add_argument("-base_path", type=str, help="Path to the DomainNet data folder")
args = parser.parse_args()

base_path = args.base_path

# List of domains/tasks
domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

for domain in domains:
    print(f"Processing {domain}...")
    
    # Paths to the txt files
    train_txt = os.path.join(base_path, f"{domain}_train.txt")
    test_txt = os.path.join(base_path, f"{domain}_test.txt")
    
    # New folder structure
    new_train_dir = os.path.join(base_path, f"{domain}_new", "train")
    new_test_dir = os.path.join(base_path, f"{domain}_new", "test")
    
    # Create new train/test folders if not exist
    os.makedirs(new_train_dir, exist_ok=True)
    os.makedirs(new_test_dir, exist_ok=True)
    
    # Function to move images
    def move_images(txt_file, target_dir):
        with open(txt_file, "r") as f:
            for line in f:
                img_path, label = line.strip().split()
                class_name = os.path.dirname(img_path).split("/")[-1]  # extract class name
                
                # Make class folder
                class_dir = os.path.join(target_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                # Source and destination path
                src = os.path.join(base_path, img_path)
                dst = os.path.join(class_dir, os.path.basename(img_path))
                
                # Move the image
                shutil.move(src, dst)
    
    # Move train and test images
    move_images(train_txt, new_train_dir)
    move_images(test_txt, new_test_dir)

print("All domains moved successfully!")

