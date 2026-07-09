from pathlib import Path
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch

from evaluate import load_model
import utils as u
import configs as cfg
from configs import config as c

def predict(img):
    m = load_model(cfg.FINAL_DIR,c)
    m.eval()

    actual = u.identify_img(img, eval=True)[1]

    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Path(img)
    img = Image.open(img)

    img = transform(img)
    img = img.unsqueeze(0)

    img = img.to(cfg.DEVICE)

    with torch.no_grad():
        out = m(img)

        p = torch.softmax(out, dim=1)

        top_probs, top_preds = torch.topk(p, k=5, dim=1)

    top_probs = top_probs.squeeze(0).tolist()
    top_preds = top_preds.squeeze(0).tolist()
    
    return actual, top_preds, top_probs

if __name__ == "__main__":
    from pathlib import Path
    import random

    test_dir = Path("data/mosquito_data/test")

    image = random.choice(list(test_dir.glob("*.jpg")))

    print(str(image))

    pred = predict(str(image))

    print("Actual:", pred[0])
    for rank, (cls, prob) in enumerate(zip(pred[1], pred[2]), start=1):
        print(f"     {rank}. {cfg.MOSQ_MAP[cls]:<30}: {100*prob:.5f}%")