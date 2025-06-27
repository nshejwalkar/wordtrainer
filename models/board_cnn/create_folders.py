import os

DATA_DIR = '../../models_data/train'

for name in os.listdir(DATA_DIR):
   path = os.path.join(DATA_DIR, name)
   if os.path.isdir(path) and name.isalpha() and len(name) == 1:
      upper_name = name.upper()
      new_path = os.path.join(DATA_DIR, upper_name)
      if name != upper_name:
         print(f"Renaming: {name} → {upper_name}")
         os.rename(path, new_path)
