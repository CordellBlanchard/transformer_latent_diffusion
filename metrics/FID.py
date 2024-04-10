from pytorch_fid.fid_score import calculate_fid_given_paths

# Paths to the directories containing real and generated images
real_images_path = 'path/to/real/images'
generated_images_path = 'path/to/generated/images'

# Calculate FID
fid_value = calculate_fid_given_paths([real_images_path, generated_images_path], 256, 'cuda', 2048)
print('FID score:', fid_value)