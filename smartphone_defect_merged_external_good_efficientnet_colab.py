# SMARTPHONE DEFECT CLASSIFICATION — MERGED REAL-IMAGE DATASET
# ================================================================
# Goal:
#   Girish dataset is the BASE.
#   Add ONLY genuine Good/Intact/Non-cracked images from:
#     1) Cracked & Intact Smartphone Images Dataset (Kaggle)
#     2) Smartphone Screen Damage Detection Dataset (Figshare)
#
#   DO NOT copy:
#     - Cracked/Broken images
#     - Oil/Scratch/Stain from external datasets
#     - Pratham/Pratul notebook inputs (they are the same Girish dataset)
#
#   Then:
#     - remove exact duplicate images using SHA-256
#     - create a NEW stratified train/validation/test split
#     - train EfficientNet-B0
#     - NO SMOTE
#     - NO synthetic augmentation
#     - class weighting is optional and OFF by default because the target
#       is approximately balanced real images per class.
#
# IMPORTANT:
#   The exact folder name inside the Kaggle Cracked & Intact dataset can
#   vary. This notebook searches for likely Whole/Intact folders and stops
#   if it cannot safely identify them. It does NOT guess.
#
#   The Figshare archive is downloaded directly and searched for
#   not_cracked folders under train/valid/test.

# ================================================================
# CELL 1 — Install packages
# ================================================================

!pip -q install kaggle scikit-learn seaborn

# ================================================================
# CELL 2 — Upload kaggle.json
# ================================================================

from google.colab import files
import os, shutil
from pathlib import Path

print("Upload kaggle.json")
uploaded = files.upload()

os.makedirs("/root/.kaggle", exist_ok=True)
shutil.copy("kaggle.json", "/root/.kaggle/kaggle.json")
os.chmod("/root/.kaggle/kaggle.json", 0o600)

print("Kaggle API configured.")

# ================================================================
# CELL 3 — Download Girish dataset
# ================================================================

GIRISH_SLUG = "girish17019/mobile-phone-defect-segmentation-dataset"

BASE = Path("/content/project_data")
GIRISH_DIR = BASE / "girish"
GIRISH_DIR.mkdir(parents=True, exist_ok=True)

!kaggle datasets download -d {GIRISH_SLUG} -p {GIRISH_DIR} --unzip

print("Girish downloaded.")

# ================================================================
# CELL 4 — Download Cracked & Intact Smartphone dataset
# ================================================================

CRACKED_INTACT_SLUG = "axondata/cracked-and-intact-smartphone-images-dataset"

CI_DIR = BASE / "cracked_intact"
CI_DIR.mkdir(parents=True, exist_ok=True)

!kaggle datasets download -d {CRACKED_INTACT_SLUG} -p {CI_DIR} --unzip

print("Cracked & Intact downloaded.")

# ================================================================
# CELL 5 — Download Figshare Smartphone Screen Damage dataset
# ================================================================
# Direct Figshare download link for the dataset archive.
#
# If Figshare changes the download endpoint, replace FIGSHARE_URL
# with the current "Download all" URL from the Figshare page:
# https://figshare.com/articles/dataset/Data_Set_Smartphone_Screen_Damage_Detection_zip/29108471

FIGSHARE_URL = "https://figshare.com/ndownloader/files/52384371"

FIG_DIR = BASE / "figshare"
FIG_DIR.mkdir(parents=True, exist_ok=True)

import requests

zip_path = FIG_DIR / "figshare_dataset.zip"

r = requests.get(FIGSHARE_URL, stream=True, timeout=120)
r.raise_for_status()

with open(zip_path, "wb") as f:
    for chunk in r.iter_content(chunk_size=1024 * 1024):
        if chunk:
            f.write(chunk)

print("Figshare archive downloaded:", zip_path)

# ================================================================
# CELL 6 — Extract Figshare archive
# ================================================================

import zipfile

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(FIG_DIR / "extracted")

print("Figshare extracted.")

# ================================================================
# CELL 7 — Inspect all downloaded folders
# ================================================================

def print_tree(root, max_depth=4):
    root = Path(root)
    print(f"\nTREE: {root}")
    for p in sorted(root.rglob("*")):
        depth = len(p.relative_to(root).parts)
        if depth <= max_depth:
            prefix = "  " * (depth - 1)
            print(prefix + ("📁 " if p.is_dir() else "📄 ") + p.name)

print_tree(GIRISH_DIR, 4)
print_tree(CI_DIR, 4)
print_tree(FIG_DIR / "extracted", 5)

# ================================================================
# CELL 8 — Helper functions
# ================================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def image_files(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

def folders_named(root, names):
    names = {x.lower() for x in names}
    return [
        p for p in Path(root).rglob("*")
        if p.is_dir() and p.name.lower() in names
    ]

# ================================================================
# CELL 9 — Locate Girish class folders
# ================================================================

GIRISH_CLASSES = ["good", "oil", "scratch", "stain"]

girish_class_dirs = {}

for cls in GIRISH_CLASSES:
    matches = folders_named(GIRISH_DIR, [cls])
    if len(matches) == 0:
        raise FileNotFoundError(f"Girish class folder not found: {cls}")
    if len(matches) > 1:
        print(f"WARNING: multiple '{cls}' folders found:")
        for m in matches:
            print(" ", m)
    girish_class_dirs[cls] = matches[0]

print("\nGirish class folders:")
for cls, p in girish_class_dirs.items():
    print(cls, "->", p)

# ================================================================
# CELL 10 — Count original Girish images
# ================================================================

print("\nGirish counts:")
for cls in GIRISH_CLASSES:
    print(f"{cls:10s}: {len(image_files(girish_class_dirs[cls]))}")

# ================================================================
# CELL 11 — Locate Cracked & Intact WHOLE / INTACT images
# ================================================================
# We ONLY want Whole/Intact.
# We explicitly reject Broken/Cracked.
#
# The code searches folder names. If more than one candidate is found,
# it stops so you can inspect instead of accidentally selecting the wrong
# folder.

whole_candidates = folders_named(
    CI_DIR,
    ["whole", "intact", "non_cracked", "non-cracked", "not_cracked", "not-cracked"]
)

# Remove nested duplicates: if a candidate is inside another candidate,
# keep the highest-level candidate.
whole_candidates = sorted(
    set(whole_candidates),
    key=lambda p: len(p.parts)
)

filtered = []
for p in whole_candidates:
    if not any(parent in p.parents for parent in filtered):
        filtered.append(p)

whole_candidates = filtered

print("\nPotential Whole/Intact folders:")
for p in whole_candidates:
    print(p, "->", len(image_files(p)), "images")

if len(whole_candidates) == 0:
    raise RuntimeError(
        "No Whole/Intact folder was safely identified. "
        "Inspect the tree in CELL 7 and set CI_GOOD_DIR manually."
    )

if len(whole_candidates) > 1:
    raise RuntimeError(
        "Multiple Whole/Intact candidates found. "
        "Inspect CELL 11 and manually choose the correct folder."
    )

CI_GOOD_DIR = whole_candidates[0]
print("\nUsing Cracked & Intact Good source:", CI_GOOD_DIR)

# ================================================================
# CELL 12 — Locate Figshare NOT-CRACKED images from ALL splits
# ================================================================
# We intentionally use:
#   train/not_cracked
#   valid/not_cracked
#   test/not_cracked
#
# because you said you want to merge the raw data first and then
# create a completely new train/validation/test split.

fig_extracted = FIG_DIR / "extracted"

not_cracked_candidates = folders_named(
    fig_extracted,
    ["not_cracked", "not-cracked", "non_cracked", "non-cracked"]
)

# Select only candidates that have train/valid/test as an ancestor.
# If multiple copies exist because of archive nesting, choose folders
# under a path containing the expected split names.

fig_good_dirs = []

for p in not_cracked_candidates:
    parts_lower = [x.lower() for x in p.parts]
    if any(x in parts_lower for x in ["train", "valid", "validation", "test"]):
        fig_good_dirs.append(p)

print("\nFigshare not-cracked folders found:")
for p in fig_good_dirs:
    print(p, "->", len(image_files(p)), "images")

if len(fig_good_dirs) == 0:
    raise RuntimeError(
        "Could not find Figshare not_cracked folders. "
        "Inspect CELL 7 and set FIG_GOOD_DIRS manually."
    )

# ================================================================
# CELL 13 — Build a source manifest BEFORE copying
# ================================================================

import pandas as pd

manifest_rows = []

# Girish
for cls in GIRISH_CLASSES:
    for p in image_files(girish_class_dirs[cls]):
        manifest_rows.append({
            "source": "girish",
            "source_class": cls,
            "final_class": cls,
            "path": str(p)
        })

# Cracked & Intact: ONLY Whole/Intact -> Good
for p in image_files(CI_GOOD_DIR):
    manifest_rows.append({
        "source": "cracked_intact",
        "source_class": "whole_or_intact",
        "final_class": "good",
        "path": str(p)
    })

# Figshare: ALL not_cracked splits -> Good
for p in fig_good_dirs:
    split_name = next(
        (x for x in p.parts if x.lower() in ["train", "valid", "validation", "test"]),
        "unknown"
    )

    for img in image_files(p):
        manifest_rows.append({
            "source": "figshare",
            "source_class": f"{split_name}_not_cracked",
            "final_class": "good",
            "path": str(img)
        })

manifest = pd.DataFrame(manifest_rows)

print("\nRAW SOURCE COUNTS:")
display(
    manifest.groupby(["source", "final_class"])
    .size()
    .unstack(fill_value=0)
)

# ================================================================
# CELL 14 — Exact duplicate removal using SHA-256
# ================================================================

import hashlib

def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

print("Computing SHA-256 hashes...")

manifest["sha256"] = manifest["path"].apply(sha256_file)

before = len(manifest)

# Keep the first copy of an exact duplicate.
# Girish is listed first, so if an external source contains an exact copy
# of a Girish image, the Girish copy is retained.

manifest = manifest.drop_duplicates(
    subset=["sha256"],
    keep="first"
).reset_index(drop=True)

after = len(manifest)

print("Images before exact-duplicate removal:", before)
print("Images after exact-duplicate removal: ", after)
print("Exact duplicates removed:              ", before - after)

# ================================================================
# CELL 15 — Final merged counts BEFORE new split
# ================================================================

print("\nFINAL UNIQUE IMAGE COUNTS:")
final_counts = manifest["final_class"].value_counts().reindex(
    ["good", "oil", "scratch", "stain"]
).fillna(0).astype(int)

display(final_counts)

# ================================================================
# CELL 16 — Optional target count for Good
# ================================================================
# We DO NOT force exactly 400 by duplicating images.
# If there are >400 unique Good images, you can either:
#   A) keep all of them (recommended), OR
#   B) randomly select exactly 400.
#
# Default: KEEP ALL UNIQUE GOOD IMAGES.
#
# Change KEEP_ONLY_N_GOOD = True only if you specifically need ~400.

KEEP_ONLY_N_GOOD = False
TARGET_GOOD = 400

if KEEP_ONLY_N_GOOD:
    good_df = manifest[manifest["final_class"] == "good"].copy()
    other_df = manifest[manifest["final_class"] != "good"].copy()

    if len(good_df) < TARGET_GOOD:
        print(
            f"Only {len(good_df)} unique Good images available; "
            f"cannot create {TARGET_GOOD} real images without duplication."
        )
    else:
        good_df = good_df.sample(
            n=TARGET_GOOD,
            random_state=42
        )
        manifest = pd.concat(
            [good_df, other_df],
            ignore_index=True
        )

print("\nCounts after optional Good selection:")
display(
    manifest["final_class"]
    .value_counts()
    .reindex(["good", "oil", "scratch", "stain"])
)

# ================================================================
# CELL 17 — Copy selected images into one merged raw dataset
# ================================================================

MERGED_RAW = BASE / "merged_raw"

if MERGED_RAW.exists():
    shutil.rmtree(MERGED_RAW)

for cls in ["good", "oil", "scratch", "stain"]:
    (MERGED_RAW / cls).mkdir(parents=True, exist_ok=True)

# Unique names prevent collisions and preserve source information.
for i, row in manifest.iterrows():

    src = Path(row["path"])
    cls = row["final_class"]
    source = row["source"]

    ext = src.suffix.lower()
    new_name = f"{source}_{i:06d}{ext}"

    dst = MERGED_RAW / cls / new_name
    shutil.copy2(src, dst)

print("Merged raw dataset created at:")
print(MERGED_RAW)

# ================================================================
# CELL 18 — Verify merged dataset
# ================================================================

print("\nMERGED RAW COUNTS:")
for cls in ["good", "oil", "scratch", "stain"]:
    print(f"{cls:10s}: {len(image_files(MERGED_RAW / cls))}")

# ================================================================
# CELL 19 — NEW stratified train/validation/test split
# ================================================================
# 70% Train / 15% Validation / 15% Test.
#
# This is a NEW split across the merged raw dataset.
# No original Figshare split boundaries are preserved.
#
# No augmentation is performed.

from sklearn.model_selection import train_test_split

split_df = []

for label, cls in enumerate(["good", "oil", "scratch", "stain"]):
    for p in image_files(MERGED_RAW / cls):
        split_df.append({
            "path": str(p),
            "class": cls,
            "label": label
        })

split_df = pd.DataFrame(split_df)

train_df, temp_df = train_test_split(
    split_df,
    test_size=0.30,
    stratify=split_df["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

print("\nTrain distribution:")
display(train_df["class"].value_counts().reindex(["good","oil","scratch","stain"]))

print("\nValidation distribution:")
display(val_df["class"].value_counts().reindex(["good","oil","scratch","stain"]))

print("\nTest distribution:")
display(test_df["class"].value_counts().reindex(["good","oil","scratch","stain"]))

# ================================================================
# CELL 20 — Create physical train/val/test folders
# ================================================================

FINAL_DATASET = BASE / "final_dataset"

if FINAL_DATASET.exists():
    shutil.rmtree(FINAL_DATASET)

for split_name in ["train", "val", "test"]:
    for cls in ["good", "oil", "scratch", "stain"]:
        (FINAL_DATASET / split_name / cls).mkdir(
            parents=True,
            exist_ok=True
        )

def copy_split(dataframe, split_name):
    for _, row in dataframe.iterrows():
        src = Path(row["path"])
        dst = FINAL_DATASET / split_name / row["class"] / src.name
        shutil.copy2(src, dst)

copy_split(train_df, "train")
copy_split(val_df, "val")
copy_split(test_df, "test")

print("Final dataset created:", FINAL_DATASET)

# ================================================================
# CELL 21 — Verify final dataset
# ================================================================

for split_name in ["train", "val", "test"]:
    print(f"\n{split_name.upper()}")
    for cls in ["good", "oil", "scratch", "stain"]:
        n = len(image_files(FINAL_DATASET / split_name / cls))
        print(f"{cls:10s}: {n}")

# ================================================================
# CELL 22 — Visual sanity check
# ================================================================

import matplotlib.pyplot as plt
from PIL import Image
import random

def show_random_images(folder, n=6, title=""):
    imgs = image_files(folder)
    n = min(n, len(imgs))

    selected = random.sample(imgs, n)

    plt.figure(figsize=(15, 8))

    for i, p in enumerate(selected):
        img = Image.open(p).convert("RGB")
        ax = plt.subplot(2, 3, i + 1)
        ax.imshow(img)
        ax.set_title(p.name[:25])
        ax.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

show_random_images(
    FINAL_DATASET / "train" / "good",
    title="Random Good Images — Training"
)

# ================================================================
# CELL 23 — PyTorch / EfficientNet setup
# ================================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

CLASS_NAMES = ["good", "oil", "scratch", "stain"]

# NO AUGMENTATION.
# Only deterministic preprocessing.
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ================================================================
# CELL 24 — Dataset class
# ================================================================

class SmartphoneDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(row["label"])

        return image, label

train_dataset = SmartphoneDataset(train_df, transform)
val_dataset = SmartphoneDataset(val_df, transform)
test_dataset = SmartphoneDataset(test_df, transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

# ================================================================
# CELL 25 — EfficientNet-B0
# ================================================================

model = efficientnet_b0(
    weights=EfficientNet_B0_Weights.DEFAULT
)

in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    len(CLASS_NAMES)
)

model = model.to(device)

print(model.classifier)

# ================================================================
# CELL 26 — Loss
# ================================================================
# Because the goal is approximately balanced real images per class,
# use normal CrossEntropyLoss by default.
#
# If final counts are still significantly imbalanced, you can turn
# USE_CLASS_WEIGHTS = True.

USE_CLASS_WEIGHTS = False

if USE_CLASS_WEIGHTS:

    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    train_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CLASS_NAMES)),
        y=train_df["label"].values
    )

    train_weights = torch.tensor(
        train_weights,
        dtype=torch.float32
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=train_weights
    )

    print("Class-weighted CrossEntropyLoss ENABLED.")
    print(train_weights)

else:

    criterion = nn.CrossEntropyLoss()

    print("Standard CrossEntropyLoss ENABLED.")

# ================================================================
# CELL 27 — Optimizer
# ================================================================

import torch.optim as optim

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)

# ================================================================
# CELL 28 — Train / validation functions
# ================================================================

import numpy as np

def train_one_epoch(model, loader):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(1)

        total_loss += loss.item() * images.size(0)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )

# ================================================================
# CELL 29 — Train
# ================================================================

EPOCHS = 15
PATIENCE = 5

best_val_loss = float("inf")
best_state = None
no_improve = 0

history = []

for epoch in range(EPOCHS):

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader
    )

    val_loss, val_acc, _, _, _ = evaluate(
        model,
        val_loader
    )

    scheduler.step(val_loss)

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    })

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} | "
        f"Train Loss={train_loss:.4f} | "
        f"Train Acc={train_acc:.4f} | "
        f"Val Loss={val_loss:.4f} | "
        f"Val Acc={val_acc:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss

        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

        no_improve = 0

    else:
        no_improve += 1

    if no_improve >= PATIENCE:
        print("Early stopping.")
        break

if best_state is not None:
    model.load_state_dict(best_state)

# ================================================================
# CELL 30 — Final test evaluation
# ================================================================

test_loss, test_acc, y_true, y_pred, y_prob = evaluate(
    model,
    test_loader
)

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_acc:.4f}")

# ================================================================
# CELL 31 — Precision / Recall / F1
# ================================================================

from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

print("Weighted Precision:", round(precision, 4))
print("Weighted Recall   :", round(recall, 4))
print("Weighted F1       :", round(f1, 4))

try:
    auc = roc_auc_score(
        y_true,
        y_prob,
        multi_class="ovr",
        average="weighted"
    )
    print("Weighted ROC-AUC  :", round(auc, 4))
except Exception as e:
    print("ROC-AUC unavailable:", e)

# ================================================================
# CELL 32 — Confusion matrix
# ================================================================

import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(len(CLASS_NAMES))
)

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — EfficientNet")
plt.tight_layout()
plt.show()

# ================================================================
# CELL 33 — Training curves
# ================================================================

history_df = pd.DataFrame(history)

plt.figure(figsize=(8, 5))
plt.plot(
    history_df["epoch"],
    history_df["train_accuracy"],
    label="Train Accuracy"
)
plt.plot(
    history_df["epoch"],
    history_df["val_accuracy"],
    label="Validation Accuracy"
)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(
    history_df["epoch"],
    history_df["train_loss"],
    label="Train Loss"
)
plt.plot(
    history_df["epoch"],
    history_df["val_loss"],
    label="Validation Loss"
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# ================================================================
# CELL 34 — Save final model
# ================================================================

MODEL_PATH = "/content/efficientnet_merged_real_images.pth"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "image_size": 224,
        "use_class_weights": USE_CLASS_WEIGHTS
    },
    MODEL_PATH
)

print("Model saved:", MODEL_PATH)

# ================================================================
# CELL 35 — Download model + merged dataset manifest
# ================================================================

manifest_path = "/content/final_image_manifest.csv"
manifest.to_csv(manifest_path, index=False)

print("Manifest:", manifest_path)

files.download(MODEL_PATH)
files.download(manifest_path)

# ================================================================
# OPTIONAL CELL 36 — Predict a new image
# ================================================================

def predict_image(model, image_path):

    model.eval()

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

    idx = probabilities.argmax().item()

    return (
        CLASS_NAMES[idx],
        float(probabilities[idx].cpu())
    )

# To use:
#
# uploaded = files.upload()
# image_path = next(iter(uploaded.keys()))
#
# prediction, confidence = predict_image(
#     model,
#     image_path
# )
#
# print("Prediction:", prediction)
# print("Confidence:", f"{confidence:.4f}")
