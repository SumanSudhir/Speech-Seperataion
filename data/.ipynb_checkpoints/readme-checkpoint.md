run :- 
chmod +x generate_data.sh  
then run : - 
./generate_data.sh --start_range 0 --stop_range 1 --type train --delete_old_data true --delete_video true

chmod +x build_audio_database.sh
then run : - 
./build_audio_database.sh --start_range 0 --stop_range 1 --num_speaker 2 --max_num_sample 50 --delete_old_data true

chmod +x build_audio_database_test.sh
then run : - 
./build_audio_database_test.sh --start_range 0 --stop_range 1 --num_speaker 2 --max_num_sample 50 --delete_old_data true

