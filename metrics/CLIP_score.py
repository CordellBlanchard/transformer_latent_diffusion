import clip
import torch
from PIL import Image
from tqdm import tqdm
import os

from tld.configs import ClipConfig


# Load the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load(ClipConfig.clip_model_name, device=device)

# Paths to the directories containing generated images and text prompts
image_dir = '../../exp4_generated_images'
prompt_dir = '../../prompts'

# Retrieve and sort image and prompt file paths
generated_image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
prompt_paths = sorted([os.path.join(prompt_dir, f) for f in os.listdir(prompt_dir) if f.endswith('.txt')])
# ensure all names match
assert all(
    os.path.splitext(os.path.basename(x).replace("generated_", ""))[0] == 
    os.path.splitext(os.path.basename(y))[0]
    for x, y in zip(generated_image_paths, prompt_paths)
), "Image and text filenames do not match."



# calculate the CLIP score for each text-image pair
similarities = []
max_length = model.context_length

for image_path, prompt_path in tqdm(zip(generated_image_paths, prompt_paths)):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    # tokenize prompt
    with open(prompt_path, 'r') as file:
        prompt = file.read().strip()

    # if prompt is too long skip (~only happens about 28 times with our test set)
    try:
        text = clip.tokenize([prompt]).to(device)
    except RuntimeError as e:
        continue

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
average_similarity = (sum(similarities) / len(similarities)) * 100
print("Average CLIP Similarity Score:", average_similarity)





