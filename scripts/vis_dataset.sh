# bash scripts/vis_dataset.sh

# dataset_path=/home/mihawk/Improved-3D-Diffusion-Policy/imitation_data_example3
dataset_path=/home/mihawk/Improved-3D-Diffusion-Policy/imitation_data6
vis_cloud=1
cd Improved-3D-Diffusion-Policy
python vis_dataset.py --dataset_path $dataset_path \
                    --use_img 1 \
                    --vis_cloud ${vis_cloud} \
                    --use_pc_color 0 \
                    --downsample 1 \