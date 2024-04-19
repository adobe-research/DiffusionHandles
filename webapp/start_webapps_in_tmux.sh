#!/bin/bash

if [ -z "$1" ]
then
  echo "Usage: source start_webapps_in_tmux.sh <base_netpath>"
  return 0
fi

if ! { [ -n "$TMUX" ]; } then
  echo "This command must be run from inside a tmux session."
  return 0
fi

source stop_webapps.sh

base_netpath=$1
current_dir=$(pwd)

conda_env_name=diffusionhandles

text2img_suffix=/text2img
text2img_port=8893
text2img_device=cuda:2

foreground_selector_suffix=/foreground_selector
foreground_selector_port=8892
foreground_selector_device=cuda:0

foreground_remover_suffix=/foreground_remover
foreground_remover_port=8891
foreground_remover_device=cuda:0

depth_estimator_suffix=/depth_estimator
depth_estimator_port=8890
depth_estimator_device=cuda:0

diffhandles_suffix=/diffhandles
diffhandles_port=8889
diffhandles_device=cuda:1

diffhandles_pipeline_suffix=/dh
diffhandles_pipeline_port=8888
diffhandles_pipeline_device=cuda:0

# Start webapps.
echo "Starting webapps in new tmux windows ..."
source run_in_tmux_window.sh text2img "cd $current_dir; conda activate $conda_env_name; python webapps/stablediff_text2img_webapp.py --port $text2img_port --netpath $base_netpath$text2img_suffix --device $text2img_device"
source run_in_tmux_window.sh foreground_selector "cd $current_dir; conda activate $conda_env_name; python webapps/langsam_segmenter_webapp.py --port $foreground_selector_port --netpath $base_netpath$foreground_selector_suffix --device $foreground_selector_device"
source run_in_tmux_window.sh foreground_remover "cd $current_dir; conda activate $conda_env_name; python webapps/lama_inpainter_webapp.py --port $foreground_remover_port --netpath $base_netpath$foreground_remover_suffix --device $foreground_remover_device"
source run_in_tmux_window.sh depth_estimator "cd $current_dir; conda activate $conda_env_name; python webapps/zoe_depth_webapp.py --port $depth_estimator_port --netpath $base_netpath$depth_estimator_suffix --device $depth_estimator_device"
source run_in_tmux_window.sh diffhandles "cd $current_dir; conda activate $conda_env_name; python webapps/diffhandles_webapp.py --port $diffhandles_port --netpath $base_netpath$diffhandles_suffix --device $diffhandles_device --debug_images --return_meshes"

# Wait for the webapps to finish starting, so the pipeline webapp can connect to them.
echo "Waiting for webapps to finish starting..."
sleep 20s 

# Start the pipeline webapp which connects to the other webapps (and requires them to have finished starting)
diffhandles_pipeline_cmd=$(printf '%s' \
  "cd $current_dir; conda activate $conda_env_name; " \
  "python webapps/diffhandles_pipeline_webapp.py --port $diffhandles_pipeline_port --netpath $base_netpath$diffhandles_pipeline_suffix " \
  "--text2img_url http://localhost:$text2img_port$base_netpath$text2img_suffix " \
  "--foreground_selector_url http://localhost:$foreground_selector_port$base_netpath$foreground_selector_suffix " \
  "--foreground_remover_url http://localhost:$foreground_remover_port$base_netpath$foreground_remover_suffix " \
  "--depth_estimator_url http://localhost:$depth_estimator_port$base_netpath$depth_estimator_suffix " \
  "--diffhandles_url http://localhost:$diffhandles_port$base_netpath$diffhandles_suffix " \
  "--device $diffhandles_pipeline_device --debug_images --return_meshes")
source run_in_tmux_window.sh diffhandles_pipeline "$diffhandles_pipeline_cmd"

echo "Diffusion Handles Pipeline WebApp should be running at http://localhost:$diffhandles_pipeline_port$base_netpath$diffhandles_pipeline_suffix"
