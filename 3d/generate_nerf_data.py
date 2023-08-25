from ursina import *
import random
import json
import shutil
import os

if os.path.exists('views/'):
    shutil.rmtree('views/')
os.mkdir('views/')
app = Ursina()

entity = Entity(model='cube')

i = 0

# Open the file for writing view information.
json_data = []
n_images = 30


def update():  # this function is called every frame
    global i
    if 2 < i < n_images + 3:  # Here you can specify the number of different views you want
        camera.position = (random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))  # random position
        camera.look_at((0, 0, 0))  # look at the origin
        camera.rotation_y += random.randint(-10, 10)  # add some random rotation around the y-axis

        # Saving view (position and rotation) and image
        view_info = {
            'position': str(camera.position)[5:-1],
            'rotation': str(camera.rotation)[5:-1],
            'image': str(app.screenshot(namePrefix=f'views/screenshot_{i-3}'))  # save screenshot and return its path
        }

        # Write the view information to the file.
        json_data.append(view_info)
    elif i >= n_images + 3:
        # Close the file after the Ursina app main loop ends.
        with open(f'views/view_info.json', 'w') as f:
            json.dump(json_data, f, indent=2)
    i += 1


app.run()
