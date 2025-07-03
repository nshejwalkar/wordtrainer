import torch, cv2, os, glob
import torch.nn as nn
import torchvision.transforms as transforms

WEIGHTS   = 'board_cnn_full.pt'
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT      = os.path.abspath('../../models_data/pics')

MAGIC = {
   'board': {'ytop': 0.42552083333333335, 'ybottom': 0.7927083333333333, 'xleft': 0.10247747747747747, 'xright': 0.8975225225225225},
}

def _get_letter_from_tile(tile):
   transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Grayscale(num_output_channels=1),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5], std=[0.5])
   ])

   class TinyCNN(nn.Module):
      def __init__(self, n_classes):
         super().__init__()
         self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                # 90 → 45
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                # 45 → 22
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU()
         )
         self.pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pool  → (64,1,1)
         self.classifier = nn.Linear(64, n_classes)

      def forward(self, x):
         x = self.features(x)
         x = self.pool(x).flatten(1)
         return self.classifier(x)

   net = TinyCNN(26).to(DEVICE)
   net.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))

   with torch.no_grad():
      if len(tile.shape) == 3:
         tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)

      # Transform to tensor
      tensor = transform(tile).unsqueeze(0).to(DEVICE)  # shape (1, 1, 90, 90)

      # Run inference
      logits = net(tensor)
      # pred_class = logits.argmax(dim=1).item()
      probs = torch.softmax(logits, dim=1)
      conf, pred_class = probs.max(dim=1)
      pred_letter = chr(ord('A') + pred_class.item())
      # print(f'LETTER: {pred_letter}, CONF: {conf.item():.3f}')

      return pred_letter
   

for picture in os.listdir(ROOT):
   picture = os.path.join(ROOT, picture)
   