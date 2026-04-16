export COMET_API_KEY=dummy
export COMET_WORKSPACE='rodyarad'
export COMET_PROJECT_NAME='lightzero'
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1


#python3 -u zoo/shapes2d/config/shapes2d_unizero_segment_config.py
#python3 -u zoo/causal_world/config/causalworld_pixels_suz_segment_config.py
#python3 -u zoo/robosuite/config/robosuite_pixels_suz_segment_config.py
#python3 -u zoo/maniskill/config/maniskill_pixels_suz_segment_config.py
#python3 -u zoo/vizdoom/config/vizdoom_unizero_segment_config.py
#python3 -u zoo/mof/config/mof_pixels_suz_segment_config.py
#python3 -u zoo/ocrl/config/ocrl_unizero_segment_config_test.py\
python3 -u zoo/vizdoom/config/vizdoom_unizero_segment_config.py
