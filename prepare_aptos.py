import os
import shutil
import pandas as pd

csv_path = "data/aptos/raw/train.csv"
images_dir = "data/aptos/raw/train_images"
output_dir = "data/aptos/train"

df = pd.read_csv(csv_path)
print(f"Total images in CSV: {len(df)}")
print(f"Class distribution:\n{df['diagnosis'].value_counts().sort_index()}\n")

for cls in range(5):
    os.makedirs(os.path.join(output_dir, str(cls)), exist_ok=True)

ok = 0
missing = 0

for _, row in df.iterrows():
    image_name = str(row["id_code"]) + ".png"
    src = os.path.join(images_dir, image_name)
    dst = os.path.join(output_dir, str(row["diagnosis"]), image_name)

    if os.path.exists(src):
        shutil.copy2(src, dst)
        ok += 1
    else:
        missing += 1

print(f"Done. Copied: {ok} | Not found: {missing}")
print(f"Output: {output_dir}/")