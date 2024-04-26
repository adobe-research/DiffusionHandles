import os
import json
from collections import OrderedDict
import glob

from jinja2 import Environment, FileSystemLoader


def generate_results_webpage(
        test_set_path: str, website_path: str, relative_image_dir: str,
        show_denoising_steps: bool = False, num_timesteps: int = None, num_optsteps: int = None):

    env = Environment(
        loader=FileSystemLoader('webpage_templates')
    )
    results_template = env.get_template('results_template.html')

    with open(test_set_path, 'r') as f:
        dataset_names = json.load(f, object_pairs_hook=OrderedDict)

    items = []

    for sample_name, transform_names in dataset_names.items():

        # one row per edit
        for transform_name in transform_names:
            # os.path.exists(f'{os.path.dirname(website_path)}/{relative_image_dir}/{sample_name}/{transform_name}_denoising_steps')
            
            items.append(dict(
                input_url=f'{relative_image_dir}/{sample_name}/input.png',
                mask_url=f'{relative_image_dir}/{sample_name}/mask.png',
                depth_url=f'{relative_image_dir}/{sample_name}/depth.png',
                bg_url=f'{relative_image_dir}/{sample_name}/bg.png',
                bg_depth_url=f'{relative_image_dir}/{sample_name}/bg_depth.png',
                inverted_url=f'{relative_image_dir}/{sample_name}/recon.png',
                edit=f'{relative_image_dir}/{sample_name}/{transform_name}.png',
                edit_disp_url=f'{relative_image_dir}/{sample_name}/{transform_name}_disparity.png',
                show_denoising_steps=show_denoising_steps,
                denoising_steps_url=f'{relative_image_dir}/{sample_name}/{transform_name}_denoising_steps/denoising_steps.html',
            ))

            if show_denoising_steps:
                denoising_steps_template = env.get_template('denoising_steps_template.html')

                # img_paths = glob.glob(f'{os.path.dirname(website_path)}/{relative_image_dir}/{sample_name}/{transform_name}_denoising_steps/step_*.png')
                # denoising_step_inds = []
                # optimization_step_inds = []
                # for img_path in img_paths:
                #     img_name = os.path.splitext(os.path.basename(img_path))[0]
                #     denoising_step_inds.append(int(img_name.split('_')[1]))
                #     optimization_step_inds.append(int(img_name.split('_')[3]))

                # num_denoising_steps = max(denoising_step_inds)+1
                # num_optimization_steps = max(optimization_step_inds)+1

                denoising_steps = []
                for denoising_step_idx in range(num_timesteps):
                    denoising_step = dict(
                        index=denoising_step_idx,
                        opt_steps=[]
                    )
                    for optimization_step_idx in range(num_optsteps+1):
                        denoising_step['opt_steps'].append(dict(
                            url=f'./step_{denoising_step_idx}_opt_{optimization_step_idx}.png'
                        ))
                    denoising_steps.append(denoising_step)

                denoising_steps_website_path = f'{os.path.dirname(website_path)}/{relative_image_dir}/{sample_name}/{transform_name}_denoising_steps/denoising_steps.html'
                os.makedirs(os.path.dirname(denoising_steps_website_path), exist_ok=True)
                with open(denoising_steps_website_path, 'w') as f:
                    f.write(denoising_steps_template.render(
                        optimization_steps=list(range(num_optsteps+1)),
                        denoising_steps=denoising_steps))

    os.makedirs(os.path.dirname(website_path), exist_ok=True)
    with open(website_path, 'w') as f:
        f.write(results_template.render(items=items))

if __name__ == '__main__':
    generate_results_webpage(
        test_set_path = 'data/photogen/photogen.json',
        website_path = 'results/photogen/photogen.html',
        relative_image_dir = '.')
