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
        --num_speaker)
        speaker="$2"
        shift 
        shift 
        ;;
        --max_num_sample)
        max_samples="$2"
        shift
        shift
        ;;
        --delete_old_data)
        delete_data_true_false="$2"
        shift
        shift
        ;;
    esac
done

cd audio 
python build_audio_database_test.py --start_range $start --stop_range $stop --num_speaker $speaker --max_num_sample $max_samples --delete_old_data $delete_data_true_false
