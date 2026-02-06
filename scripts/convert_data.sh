# bash convert_data.sh


save_img=1
save_depth=0


demo_path=/home/mihawk/Improved-3D-Diffusion-Policy/raw3
save_path=/home/mihawk/Improved-3D-Diffusion-Policy/imitation_data6

python convert_demos.py --demo_dir ${demo_path} \
                                --save_dir ${save_path} \
                                --save_img ${save_img} \
                                --save_depth ${save_depth} \
