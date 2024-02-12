import os
import json
from collections import OrderedDict

from jinja2 import Environment, FileSystemLoader


def generate_results_webpage(test_set_path, website_path, relative_image_dir):

    env = Environment(
        loader=FileSystemLoader('results_template.html')
    )

    template = env.get_template('')

    with open(test_set_path, 'r') as f:
        test_set_info = json.load(f, object_pairs_hook=OrderedDict)

    items = []
    for sample_name in test_set_info.keys():

        items.append(dict(
            input_url=f'{relative_image_dir}/{sample_name}/input.png',
            mask_url=f'{relative_image_dir}/{sample_name}/mask.png',
            depth_url=f'{relative_image_dir}/{sample_name}/depth.png',
            bg_url=f'{relative_image_dir}/{sample_name}/bg.png',
            bg_depth_url=f'{relative_image_dir}/{sample_name}/bg_depth.png',
            inverted_url=f'{relative_image_dir}/{sample_name}/recon.png',
            edit1_disp_url=f'{relative_image_dir}/{sample_name}/edit_000_disparity.png',
            edit1_depth_raw_url=f'{relative_image_dir}/{sample_name}/edit_000_depth_raw.png',
            edit1=f'{relative_image_dir}/{sample_name}/edit_000.png',
            edit2=f'{relative_image_dir}/{sample_name}/edit_001.png',
            edit3=f'{relative_image_dir}/{sample_name}/edit_002.png',
        ))

    with open(website_path, 'w') as f:
        f.write(template.render(items=items))

if __name__ == '__main__':
    generate_results_webpage(
        test_set_path = 'data/test_set.json',
        website_path = 'results/results.html',
        relative_image_dir = '.')
