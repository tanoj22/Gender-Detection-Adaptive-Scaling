import os, re, random
from typing import List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms

FNAME_RE = re.compile(r"^(\d+)_([01])_([0-4])_.*\.(jpg|jpeg|png)$", re.IGNORECASE)

class UTKFace(Dataset):
    def __init__(self, root: str, img_size: int = 224, sensitive_attr: str = "race",
                 age_bins: List[int] = (0,18,30,45,60,120), augment: bool = True):
        self.root = root
        self.sensitive_attr = sensitive_attr
        self.age_bins = list(age_bins)
        self.samples: List[Tuple[str, int, int, int]] = []
        for fname in os.listdir(root):
            m = FNAME_RE.match(fname)
            if not m: continue
            age = int(m.group(1)); gender = int(m.group(2)); race = int(m.group(3))
            self.samples.append((os.path.join(root, fname), gender, race, age))

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406],
                                     std=[0.229,0.224,0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406],
                                     std=[0.229,0.224,0.225]),
            ])

    def __len__(self): return len(self.samples)

    def _age_to_bin(self, age: int) -> int:
        for i in range(len(self.age_bins)-1):
            if self.age_bins[i] <= age < self.age_bins[i+1]: return i
        return len(self.age_bins)-2

    def __getitem__(self, idx: int):
        path, gender, race, age = self.samples[idx]
        img = self.transform(Image.open(path).convert("RGB"))
        if self.sensitive_attr == "race":
            group_id = race
        elif self.sensitive_attr == "agebin":
            group_id = self._age_to_bin(age)
        else:
            raise ValueError("Invalid sensitive_attr")
        return img, gender, group_id, idx

def balanced_subset_by_gender(ds: UTKFace, per_class: int = 3000, seed: int = 1337) -> Subset:
    random.seed(seed)
    idx_male = [i for i, s in enumerate(ds.samples) if s[1] == 0]
    idx_female = [i for i, s in enumerate(ds.samples) if s[1] == 1]
    random.shuffle(idx_male); random.shuffle(idx_female)
    sel = idx_male[:per_class] + idx_female[:per_class]
    random.shuffle(sel)
    return Subset(ds, sel)
