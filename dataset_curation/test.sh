pip install open3d
pip install blenderproc
install pytorch3d

python ldi_distributed_gso.py \
    --input_models_path /media/rli/Expansion/DLEE/research/datasets/gso/files \
	--object_path_file gso/gso_object_list.json \
    --num_images_per_ele 2 \
    --ele_angles 0 30 60 \
    --azimuths_offset_angle 5 \
    --camera_dist_low 1.6 \
    --camera_dist_high 2.2 \
    --render_first_n_scenes 4 \
    --save_res_path /home/rli/Research/res/lari_dataset_gen/gso \
    --render_timeout 2000 \
    --blender_path /home/rli/Downloads/softwares/blender-4.2.20-linux-x64



# 1. downloading cc_textures for textured wall/floor/ceil rendering
blenderproc download cc_textures /ibex/project/c2302/xmap/dataset/3d_front/cc_texture
python ldi_distributed_front3d.py \
    --input_models_path /ibex/project/c2302/xmap/dataset/3d_front/3D-FRONT \
	--object_path_file /ibex/ai/home/lir0d/Research/xmap/dataset_curation/ldi_gen/front3d/3d_front_house_list_dlee.json \
    --num_renders 2 \
    --render_first_n_scenes 2 \
    --save_res_path /ibex/project/c2302/xmap/dataset/3d_front/3d_front_render \
    --ldi_layers 10 \
    --online_sanity_check 0 \
    --blender_path /home/rli/Downloads/softwares/blender-4.2.20-linux-x64


python ldi_distributed_objaverse.py \
    --input_models_path /media/rli/Expansion/DLEE/research/datasets/objaverse/objaverse_object_lgm \
	--object_path_file /home/rli/Research/code/lari/dataset_curation/objaverse/lgm_v1_16K.json \
    --blender_path /home/rli/Downloads/softwares/blender-4.2.20-linux-x64 \
    --num_renders 2 \
    --render_first_n_scenes 2 \
    --save_res_path /home/rli/Research/res/lari_dataset_gen/objaverse \
    --ldi_layers 15 \
    --online_sanity_check 1


python ldi_distributed_scrream.py \
    --input_models_path /media/rli/Expansion/DLEE/research/datasets/scrream/scrream \
	--object_path_file /home/rli/Research/code/lari/dataset_curation/scrream/scrream_datalist.json \
    --render_first_n_scenes 1 \
    --render_timeout 2000 \
    --num_cpu_for_each_process 5



python ldi_distributed_scannetpp.py \
    --input_models_path /media/rli/Expansion/DLEE/research/datasets/scannetpp_v2/data \
	--object_path_file /home/rli/Research/code/lari/dataset_curation/scannetpp/scannetpp_48k.json \
    --render_first_n_scenes 1 \
    --render_timeout 2000 \
    --num_cpu_for_each_process 5 \
    --point_priority_thres 10000

    