import os
import json
from collections import OrderedDict
import glob

from jinja2 import Environment, FileSystemLoader


def generate_results_webpage(test_set_path, website_path, relative_image_dir):

    env = Environment(
        loader=FileSystemLoader('results_template.html')
    )

    template = env.get_template('')

    with open(test_set_path, 'r') as f:
        dataset_names = json.load(f, object_pairs_hook=OrderedDict)

    items = []
    for sample_name, transform_names in dataset_names.items():

        # one row per edit
        for transform_name in transform_names:
            items.append(dict(
                input_url=f'{relative_image_dir}/{sample_name}/input.png',
                mask_url=f'{relative_image_dir}/{sample_name}/mask.png',
                depth_url=f'{relative_image_dir}/{sample_name}/depth.png',
                bg_url=f'{relative_image_dir}/{sample_name}/bg.png',
                bg_depth_url=f'{relative_image_dir}/{sample_name}/bg_depth.png',
                inverted_url=f'{relative_image_dir}/{sample_name}/recon.png',
                edit=f'{relative_image_dir}/{sample_name}/{transform_name}.png',
                edit_disp_url=f'{relative_image_dir}/{sample_name}/{transform_name}_disparity.png',
                edit_depth_raw_url=f'{relative_image_dir}/{sample_name}/{transform_name}_depth_raw.png',
            ))

    with open(website_path, 'w') as f:
        f.write(template.render(items=items))

if __name__ == '__main__':
    generate_results_webpage(
        test_set_path = 'data/photogen/photogen.json',
        website_path = 'results/photogen/photogen.html',
        relative_image_dir = '.')
