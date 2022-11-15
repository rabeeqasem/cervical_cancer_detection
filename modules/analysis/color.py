
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

color_map = {
    'violet': [148, 0, 211],
    'blue': [0, 0, 255],
    'green': [0, 255, 0],
    'yellow': [255, 255, 0],
    'orange': [255, 127, 0],
    'red': [255, 0, 0],

    'black': [0, 0, 0],
    'white': [255, 255, 255]
}

def _classify_color(v1):
    min_dist = -1
    for c, v in color_map.items():
        dist = abs(np.linalg.norm(v1 - v))
        if min_dist == -1 or dist < min_dist:
            min_dist = dist
            color = c
    return color

def extract_image_colors(image_path: str):
    image = Image.open(image_path)
    img_array = np.array(image)
    img_array = img_array.reshape(img_array.shape[0] * img_array.shape[1], img_array.shape[2])

    pixel_df = pd.DataFrame(img_array)

    colors = []
    alt_r = []
    alt_g = []
    alt_b = []
    for i, row in pixel_df.iterrows():
        c = _classify_color(row.values)
        colors.append(c)
        c_rgb = color_map[c]
        alt_r.append(c_rgb[0])
        alt_g.append(c_rgb[1])
        alt_b.append(c_rgb[2])
        print(f'Done: {i}/{len(pixel_df)}', end='\r')
        
    pixel_df['color'] = colors
    pixel_df['Alt_R'] = alt_r
    pixel_df['Alt_G'] = alt_g
    pixel_df['Alt_B'] = alt_b

    vis_df = pixel_df.groupby(['color']).count()
    # vis_df = (pixel_df.groupby(['color']).count() / pixel_df.count()) * 100
    vis_df = vis_df.iloc[:, [0]]
    vis_df.reset_index(inplace=True)

    sum_df = vis_df[0].sum()

    palette = np.zeros([sum_df, 400, 3])
    total_left = 0
    for color in vis_df['color'].unique():
        temp = vis_df[vis_df['color'] == color]
        color_rgb = color_map[color]
        palette[total_left:total_left + temp[0].values[0], :] = color_rgb
        total_left += temp[0].values[0]

    # Create Altered Image
    alt_img = np.zeros(img_array.shape)
    for i, row in pixel_df.iterrows():
        alt_img[i] = [row['Alt_R'], row['Alt_G'], row['Alt_B']]

    alt_img = alt_img.reshape(image.height, image.width, 3)

    # Visualize
    fig, ax = plt.subplots(1, 3, figsize=(15, 8))
    
    ax[0].imshow(image)
    ax[1].imshow(alt_img)
    ax[2].imshow(palette)

    ax[0].set_title('Original Image')
    ax[1].set_title('Altered Image')

    ax[0].xaxis.set_visible(False)
    ax[0].yaxis.set_visible(False)
    ax[1].xaxis.set_visible(False)
    ax[1].yaxis.set_visible(False)
    
    ax[2].xaxis.set_visible(False)

    plt.show()
