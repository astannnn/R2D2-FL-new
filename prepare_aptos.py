import os
import shutil
import pandas as pd

csv_path    = "data/aptos/train.csv"
images_dir  = "data/aptos/train_images"
output_dir  = "data/aptos/train"

df = pd.read_csv(csv_path)
print(f"Total images in CSV: {len(df)}")
print(f"Class distribution:\n{df['diagnosis'].value_counts().sort_index()}\n")

for cls in range(5):
    os.makedirs(os.path.join(output_dir, str(cls)), exist_ok=True)

ok = 0
missing = 0

for _, row in df.iterrows():
    src = os.path.join(images_dir, row["id_code"] + ".png")
    dst = os.path.join(output_dir, str(row["diagnosis"]), row["id_code"] + ".png")

    if os.path.exists(src):
        shutil.copy2(src, dst)
        ok += 1
    else:
        missing += 1

print(f"Done. Copied: {ok} | Not found: {missing}")
print(f"Output: {output_dir}/")
