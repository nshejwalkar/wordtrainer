import os
import shutil

CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')

if os.path.exists(CHECKPOINTS_DIR):
   for filename in os.listdir(CHECKPOINTS_DIR):
      file_path = os.path.join(CHECKPOINTS_DIR, filename)
      try:
         if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
         elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
      except Exception as e:
         print(f'Failed to delete {file_path}. Reason: {e}')
   print("All pths deleted from checkpoints folder.")
else:
   print("Checkpoints folder does not exist.")