for arg in "$@"
do
    case $arg in
        --start_range)
        start="$2"
        shift 
        shift 
        ;;
        --stop_range)
        stop="$2"
        shift 
        shift 
        ;;
        --type)
        train_test="$2"
        shift 
        shift 
        ;;
        --delete_old_data)
        delete_data_true_false="$2"
        shift
        shift
        ;;
        --delete_video)
        true_false="$2"
        shift
        shift
        ;;
    esac
done

cd audio 
# python audio_downloader.py --start_range $start --stop_range $stop --type $train_test --delete_old_data $delete_data_true_false
# python audio_norm.py --type $train_test --delete_old_data $delete_data_true_false

cd ../video

# python video_download.py --start_range $start --stop_range $stop --type $train_test --delete_old_data $delete_data_true_false --delete_video $true_false
# python MTCNN_detect.py --start_range $start --stop_range $stop --type $train_test --delete_old_data $delete_data_true_false

cd ../Face_net
# python gen_face_emb.py --type $train_test --delete_old_data $delete_data_true_false