import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import json
from config import PRODUCTS_CSV, OUTFITS_CSV, IMAGES_DIR, CATEGORY_ROLES

class DataLoader:
    def __init__(self):
        self.products_df = None
        self.outfits_df  = None
        self._load()

    def _load(self):
        self.products_df = pd.read_csv(PRODUCTS_CSV)
        self.outfits_df  = pd.read_csv(OUTFITS_CSV)
        self._clean_products()
        self._clean_outfits()

    def _clean_products(self):
        df = self.products_df
        # Normalise string columns
        for col in ["gender","wear_type","category","category_label","occasion"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
        # Fill missing
        df["tags"]        = df["tags"].fillna("")
        df["description"] = df["description"].fillna("")
        df["rating"]      = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)
        df["rating_count"]= pd.to_numeric(df["rating_count"], errors="coerce").fillna(0)
        df["price_inr"]   = pd.to_numeric(df["price_inr"], errors="coerce").fillna(0.0)
        # Assign role
        df["role"] = df["category"].apply(self._assign_role)
        self.products_df = df

    def _clean_outfits(self):
        df = self.outfits_df
        for col in ["gender","wear_type","occasion","theme"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
        self.outfits_df = df

    def _assign_role(self, category: str) -> str:
        cat = str(category).lower().strip()
        for role, cats in CATEGORY_ROLES.items():
            if any(c in cat or cat in c for c in cats):
                return role
        return "other"

    def get_product_by_id(self, product_id: str):
        row = self.products_df[self.products_df["id"] == product_id]
        return row.iloc[0].to_dict() if not row.empty else None

    def get_products_by_role(self, role: str, gender: str = None, occasion: str = None):
        df = self.products_df[self.products_df["role"] == role]
        if gender and gender.lower() not in ["all","unisex",""]:
            df = df[df["gender"] == gender.lower()]
        if occasion:
            df = df[df["occasion"].str.contains(occasion.lower(), na=False)]
        return df

    def get_image_path(self, product_id: str) -> Path:
        row = self.get_product_by_id(product_id)
        if row and "image" in row:
            img_path = IMAGES_DIR.parent / row["image"]
            if img_path.exists():
                return img_path
        # fallback: search all subfolders
        raw_id = product_id.split("_")[-1]
        for ext in [".jpg",".jpeg",".png",".webp"]:
            for subfolder in IMAGES_DIR.iterdir():
                if subfolder.is_dir():
                    candidate = subfolder / f"{raw_id}{ext}"
                    if candidate.exists():
                        return candidate
        return None

    def load_image(self, product_id: str, size=(300,300)) -> Image.Image:
        path = self.get_image_path(product_id)
        if path:
            img = Image.open(path).convert("RGB")
            img.thumbnail(size, Image.LANCZOS)
            return img
        return None

    def build_product_text(self, row) -> str:
        parts = [
            row.get("name",""),
            row.get("brand",""),
            row.get("category_label",""),
            row.get("wear_type",""),
            row.get("occasion",""),
            row.get("description",""),
            row.get("tags","").replace(";"," "),
        ]
        return " | ".join(p for p in parts if p and p != "nan")

    def get_outfit_products(self, outfit_id: str) -> dict:
        row = self.outfits_df[self.outfits_df["outfit_id"] == outfit_id]
        if row.empty:
            return {}
        row = row.iloc[0]
        result = {}
        for role, id_col in [("hero","hero_id"),("second","second_id"),
                              ("layer","layer_id"),("footwear","footwear_id"),
                              ("accessory_1","accessory_1_id"),("accessory_2","accessory_2_id")]:
            val = row.get(id_col,"")
            if val and str(val) not in ["nan","None",""]:
                product = self.get_product_by_id(str(val).strip())
                if product:
                    result[role] = product
        return result

    def get_dataset_stats(self) -> dict:
        df = self.products_df
        return {
            "total_products":    len(df),
            "total_outfits":     len(self.outfits_df),
            "gender_dist":       df["gender"].value_counts().to_dict(),
            "occasion_dist":     df["occasion"].value_counts().to_dict(),
            "category_dist":     df["category"].value_counts().to_dict(),
            "role_dist":         df["role"].value_counts().to_dict(),
            "wear_type_dist":    df["wear_type"].value_counts().to_dict(),
            "price_stats":       {
                "min":  float(df["price_inr"].min()),
                "max":  float(df["price_inr"].max()),
                "mean": float(df["price_inr"].mean()),
            },
            "avg_rating":        float(df["rating"].mean()),
        }
