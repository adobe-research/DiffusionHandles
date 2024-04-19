#!/bin/bash
pkill -f "diffhandles_pipeline_webapp.py"
pkill -f "diffhandles_webapp.py"
pkill -f "zoe_depth_webapp.py"
pkill -f "lama_inpainter_webapp.py"
pkill -f "langsam_segmenter_webapp.py"
pkill -f "stablediff_text2img_webapp.py"
