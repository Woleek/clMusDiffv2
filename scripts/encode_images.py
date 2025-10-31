import argparse
import hashlib
import os
import pickle

from diffusers import ConfigMixin, ModelMixin
import torch
from PIL import Image as PILImage
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from datasets import load_dataset, load_from_disk, Image as DatasetImage
from tqdm.auto import tqdm

from dotenv import load_dotenv
load_dotenv()

class ImageEncoder(ModelMixin, ConfigMixin):
    def __init__(self, image_processor, encoder_model, device_local=None):
        super().__init__()
        self.processor = image_processor
        self.encoder = encoder_model
        self.device_local = device_local or next(encoder_model.parameters()).device
        self.img_loader = DatasetImage(decode=True)
        
    def forward(self, x):
        x = self.encoder(x)
        return x
        
    @torch.no_grad()
    def encode(self, images):
        self.eval()
        x = self.processor(images, return_tensors="pt")["pixel_values"].to(self.device_local)
        y = self(x).last_hidden_state
        embeddings = y[:, 0, :]
        return embeddings

def main(args):
    processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14')
    extractor = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-large-patch14')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = extractor.to(device)
    image_encoder = ImageEncoder(processor, extractor, device_local=device)
    
    if args.dataset_name_or_path is not None:
        if os.path.exists(args.dataset_name_or_path):
            dataset = load_from_disk(args.dataset_name_or_path)["train"]
        else:
            dataset = load_dataset(
                args.dataset_name_or_path,
                token=os.getenv("HF_TOKEN"),
                split="train",
            )
            
    encodings = {}
    for idx in tqdm(range(0, len(dataset), args.batch_size)):
        batch = dataset[idx:idx + args.batch_size]
        images = batch["image"]
        names = batch['sample_name']
        
        embeddings = image_encoder.encode(images).detach().cpu()

        for sample_name, embedding in zip(names, embeddings):
            encodings[sample_name] = embedding.numpy()
                
    with open(args.output_file, "wb") as f:
        pickle.dump(encodings, f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create pickled image encodings for dataset of image files.")
    parser.add_argument("--dataset_name_or_path", type=str, default="Woleek/Img2Spec")
    parser.add_argument("--output_file", type=str, default="data/encodings.p")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)