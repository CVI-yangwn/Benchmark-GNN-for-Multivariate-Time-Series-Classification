source activate gnn
time=$(date "+%Y%m%d-%H%M%S")
cd /data/yangwennuo/code/MTSC/MTSC-Graph-benchmarking

pids=()  # store child pids of python to kill them all
trap 'echo "Caught signal; killing subprocesses..."; for pid in "${pids[@]}"; do echo ${pids}; kill ${pid} || true; done; echo "kill all and exit"; exit' SIGINT SIGTERM

##---------------- param --------------------
# datasets=("ArticularyWordRecognition" "AtrialFibrillation" "BasicMotions" "Cricket" 
#           "DuckDuckGeese" "EigenWorms" "Epilepsy" "EthanolConcentration" 
#           "ERing" "FaceDetection" "FingerMovements" "HandMovementDirection" 
#           "Handwriting" "Heartbeat" "Libras" "LSST" "MotorImagery" 
#           "NATOPS" "PenDigits" "PEMS-SF" "PhonemeSpectra" "RacketSports" 
#           "SelfRegulationSCP1" "SelfRegulationSCP2" "StandWalkJump" "UWaveGestureLibrary")
datasets=("PhonemeSpectra")
# "mutual_information" "abspearson" "complete" "diffgraphlearn" "phase_locking_value"
adjs=("mutual_information")
# "raw" "differential_entropy" "power_spectral_density"
nodes=("raw")
# "chebnet" "gat" "gcn" "megat" "stgcn"
gnns=("megat")
# 42 152 310
seeds=(42)
# 0 1 2 3 4 5 6 7
gpu=0

trainLog="../train"${gpu}".log"
## ---------------- log path ------------------
## four args: 1->gnn 2->adj 3->node 4->seed
create_log_file() {
    logFile=../logs/train/$dataset/$1-$2-$3-$4.log
    parentDir=$(dirname "$logFile")
    if [[ ! -d "$parentDir" ]]; then
        echo "didn't exist direction $parentDir and creating..." >> train.log
        mkdir -p "$parentDir"
        echo "DONE" >> train.log
    fi
}
create_error_file() {
    parentDir=$(dirname "../logs/error/$dataset/$1-$2-$3-$4.log")
    if [[ ! -d "$parentDir" ]]; then
        mkdir -p "$parentDir"
    fi
}
## ----------------- config path -----------------
# only one arg: configuration file path for dataset
get_all_file_paths() {
    file_paths_list=$(find "$1" -type f)
    # for filter in "${filters[@]}"; do
    #     file_paths_list=$(echo "$file_paths_list" | grep -E $filter)
    # done
}
get_all_file_paths "./config/new/$dataset"
for dataset in "${datasets[@]}"; do
for adj in "${adjs[@]}"; do
    for gnn in "${gnns[@]}"; do
        for node in "${nodes[@]}"; do
            for seed in "${seeds[@]}"; do
                create_log_file $gnn $adj $node $seed
                create_error_file $gnn $adj $node $seed
                echo "training and logging file $logFile" >> train.log
                OMP_NUM_THREADS=8 python -u main.py \
                    --log_path $logFile \
                    --conf_file ./config/new/$dataset/$gnn-$adj-$node.yml \
                    --save_path ./models \
                    --gpu $gpu \
                    --seed $seed \
                    > "../logs/error/$dataset/$gnn-$adj-$node-$seed.log" 2>&1 &
                waitPID=$!
                pids+=($waitPID)
                wait $waitPID
            done
        done
    done
done
done