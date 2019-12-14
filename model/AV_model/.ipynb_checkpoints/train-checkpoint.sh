for arg in "$@"
do
    case $arg in
        --batch_size)
        batchsize="$2"
        shift 
        shift 
        ;;
        --LR)
        lr="$2"
        shift 
        shift 
        ;;
        --number_of_people)
        N_of_people="$2"
        shift
        shift
        ;;
        --restart_training)
        true_false="$2"
        shift
        shift
        ;;
        --path)
        path="$2"
        shift
        shift
        ;;
    esac
done

python train.py --batch_size $batchsize --LR $lr --number_of_people $N_of_people --restart_training $true_false --path $path 
