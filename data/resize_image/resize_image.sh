python create_file_list.py --input_dir=/media/yi/DATA/data-orig/robot --output_file=robot_file_list
python resize_image.py --input_dir=/media/yi/DATA/data-orig/robot --output_dir=/home/yi/code/video_motion/robot-64 --size=64 --file_list=robot_file_list
python resize_image_mp.py --input_dir=/media/yi/DATA/data-orig/robot --output_dir=/home/yi/code/video_motion/robot-64 --size=64 --file_list=robot_file_list


# python create_file_list.py --input_dir=/home/yi/Downloads/mpii_human_pose_v1_sequences --output_file=mpii_file_list
# python resize_image.py --input_dir=/home/yi/Downloads/mpii_human_pose_v1_sequences --output_dir=/home/yi/code/video_motion/mpii-128 --size=128 --file_list=mpii_file_list
# python resize_image_mp.py --input_dir=/home/yi/Downloads/mpii_human_pose_v1_sequences --output_dir=/home/yi/code/video_motion/mpii-128 --size=128 --file_list=mpii_file_list

