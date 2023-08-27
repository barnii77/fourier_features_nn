import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os
from PIL import Image
import wandb

from fourier_layer import FourierLayer  # required for loading the model


def factorize_with_smallest_difference(n):
    factors = []

    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            factors.append((i, n // i))

    best_factors = min(factors, key=lambda pair: abs(pair[0] - pair[1]))

    return best_factors


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='views')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--n-frames', type=int)
parser.add_argument('--wandb', type=str, default=None, help="name of wandb project to log this to")
args = parser.parse_args()

assert not args.cuda or torch.cuda.is_available(), "Cuda is not available. Make sure you have pytorch installed with cuda or run this file without --cuda to use cpu instead."

model = torch.load('model.ckpt')

# get the image size (=resolution)
json_fn = None
for fn in os.listdir(f"3d/{args.path}/"):
    if fn.endswith('.json'):
        assert json_fn is None, "More than one json file in data directory. Remove all but the necessary one."
        json_fn = fn

assert json_fn is not None, "Could not find a json file containing dataset info"

with open(f"3d/{args.path}/{json_fn}") as f:
    data_json = json.load(f)

img_path = data_json[0]['image']
resolution = Image.open(img_path).size

grid_in = torch.stack(
    [
        torch.linspace(-1, 1, resolution[0]).unsqueeze(-1).repeat_interleave(resolution[1], dim=1),
        torch.linspace(-1, 1, resolution[1]).unsqueeze(0).repeat_interleave(resolution[0], dim=0)
    ],
    dim=-1
)
if args.cuda:
    model = model.to('cuda')
    grid_in = grid_in.to('cuda')


# Empty function for feeding data through the model
def predict_view(position, rotation):
    model_in = (torch.tensor(position + rotation, dtype=torch.float32)
                .view((1, 1, -1))
                .repeat_interleave(resolution[0], dim=0)
                .repeat_interleave(resolution[1], dim=1))
    if args.cuda:
        model_in = model_in.to('cuda')
    model_in = torch.concat([model_in, grid_in], dim=-1)
    model_in = model_in.unsqueeze(0)
    model_out = (model(model_in) + 1) / 2
    del model_in
    out = model_out.squeeze(0).to('cpu').detach().transpose(1, 0).numpy()
    del model_out
    return out


if args.wandb is not None:
    wandb.login(anonymous='allow')
    wandb.init(
        project=args.wandb
    )

# Parameters for the animation
num_frames = args.n_frames
radius = 5.0
angle_steps = np.linspace(0, 2 * np.pi, num_frames)

# Initialize figure and 3D axes
num_rows, num_cols = factorize_with_smallest_difference(num_frames)
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5))

# List to store frames
animation_frames = []

# Generate camera positions and rotations
camera_positions = [
    [radius * np.cos(angle), radius * np.sin(angle), 3.0]
    for angle in angle_steps
]
camera_rotations = [
    [0, 0, np.arctan2(-position[1], -position[0])]
    for position in camera_positions
]

wandb_image_table = wandb.Table(columns=["frames"])  # only used if wandb project name was provided using --wandb

# Generate images using the model for each camera position and rotation
for frame in range(num_frames):
    camera_position = camera_positions[frame]
    camera_rotation = camera_rotations[frame]

    # Feed camera data through the model and get the generated image
    generated_image = predict_view(camera_position, camera_rotation)
    if args.wandb is not None:
        wandb_img = wandb.Image(generated_image)
        wandb_image_table.add_row(wandb_img)

    # Display the generated image in the respective subplot
    if num_frames > 1:
        ax = axes[frame // num_cols, frame % num_cols]
    else:
        ax = axes  # Handle the case when there's only one frame

    ax.imshow(generated_image)
    ax.set_title(f'Frame {frame + 1}')
    ax.axis('off')

if args.wandb is not None:
    wandb.log({'animation/images': wandb_image_table})
    wandb.finish()

# Adjust layout to prevent overlapping titles and labels
plt.tight_layout()

# Show the plot
plt.show()
