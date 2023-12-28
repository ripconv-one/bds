from matplotlib import pyplot as plt
import numpy as np
import h5py
from flask import Flask, render_template, url_for

dataset = h5py.File('C:/Users/PARTHIN/BDS/bdsvenv/3dshapes.h5', 'r')
print(dataset.keys())
images = dataset['images']
labels = dataset['labels']
image_shape = images.shape[1:]
label_shape = labels.shape[1:]
n_samples = labels.shape[0]  # 10*10*10*8*4*15=480000

_FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
_NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10,
                          'scale': 8, 'shape': 4, 'orientation': 15}

def get_index(factors):
    indices = 0
    base = 1
    for factor, name in reversed(list(enumerate(_FACTORS_IN_ORDER))):
        indices += factors[factor] * base
        base *= _NUM_VALUES_PER_FACTOR[name]
    return indices

def sample_random_batch(batch_size):
    indices = np.random.choice(n_samples, batch_size)
    ims = []
    for ind in indices:
        im = images[ind]
        im = np.asarray(im)
        ims.append(im)
    ims = np.stack(ims, axis=0)
    ims = ims / 255.
    ims = ims.astype(np.float32)
    return ims.reshape([batch_size, 64, 64, 3])

def sample_batch(batch_size, fixed_factor, fixed_factor_value):
    factors = np.zeros([len(_FACTORS_IN_ORDER), batch_size], dtype=np.int32)
    for factor, name in enumerate(_FACTORS_IN_ORDER):
        num_choices = _NUM_VALUES_PER_FACTOR[name]
        factors[factor] = np.random.choice(num_choices, batch_size)
    factors[fixed_factor] = fixed_factor_value
    indices = get_index(factors)
    ims = []
    for ind in indices:
        im = images[ind]
        im = np.asarray(im)
        ims.append(im)
    ims = np.stack(ims, axis=0)
    ims = ims / 255.
    ims = ims.astype(np.float32)
    return ims.reshape([batch_size, 64, 64, 3])

def show_images_grid(imgs_, num_images=25):
    ncols = int(np.ceil(num_images**0.5))
    nrows = int(np.ceil(num_images / ncols))
    _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
    axes = axes.flatten()

    for ax_i, ax in enumerate(axes):
        if ax_i < num_images:
            ax.imshow(imgs_[ax_i], cmap='Greys_r', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')

batch_size = 25
img_batch = sample_random_batch(batch_size)

show_images_grid(img_batch)


print("Choose a factor:")
for i, factor in enumerate(_FACTORS_IN_ORDER):
    print(f"{i + 1}. {factor}")

choice = int(input("Enter the number corresponding to your choice: "))
if 1 <= choice <= len(_FACTORS_IN_ORDER):
    fixed_factor_str = _FACTORS_IN_ORDER[choice - 1]
else:
    print("Invalid choice. Using the default factor.")
    fixed_factor_str = _FACTORS_IN_ORDER[0]


fixed_factor_value = int(input(f"Enter the value for {fixed_factor_str} (0 to {_NUM_VALUES_PER_FACTOR[fixed_factor_str] - 1}): "))
fixed_factor = _FACTORS_IN_ORDER.index(fixed_factor_str)
img_batch = sample_batch(batch_size, fixed_factor, fixed_factor_value)

show_images_grid(img_batch)

#plt.savefig(r'C:\Users\PARTHIN\BDS\bdsvenv\static\plot.png')
plt.savefig(r'static\plot.png')

app = Flask(__name__)

@app.route('/')
def home():
    img_path = url_for('static', filename='plot.png')
    return render_template(r'index.html', img_path=img_path)

if __name__ == '__main__':
    app.run()
