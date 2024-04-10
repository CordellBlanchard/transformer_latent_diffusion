import clip
import torch
from PIL import Image

from tld.configs import ClipConfig


# Load the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load(ClipConfig.clip_model_name, device=device)

# get generated images and prompts TODO
generated_image_paths = ['path/to/generated/image1.png', 'path/to/generated/image2.png']  # Your generated image paths
prompts = ["A description for the first image", "A description for the second image"]  # Your prompts

# calculate the CLIP score for each text-image pair
similarities = []

for image_path, prompt in zip(generated_image_paths, prompts):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    text = clip.tokenize([prompt]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    # Normalize features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Calculate similarity
    similarity = (image_features @ text_features.T).cpu().numpy().flatten()[0]
    similarities.append(similarity)

# Calculate the average similarity for a simple aggregate measure
average_similarity = sum(similarities) / len(similarities)
print("Average CLIP Similarity Score:", average_similarity)





