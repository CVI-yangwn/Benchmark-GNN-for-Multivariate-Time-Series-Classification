source activate gnn
time=$(date "+%Y%m%d-%H%M%S")
cd /data/yangwennuo/code/MTSC/MTSC-Graph-benchmarking

##---------------- param --------------------
# FingerMovements RacketSports DuckDuckGeese
dataset="DuckDuckGeese"
# "mutual_information" "abspearson" "complete" "diffgraphlearn" "phase_locking_value"
adjs=("abspearson")
# "raw" "differential_entropy" "power_spectral"
nodes=("raw")
# "chebnet" "gat" "gcn" "megat" "stgcn"
gnns=("chebnet")
# 42 152 310
seeds=(42)
# 0 1 2 3 4 5 6 7
gpus=(0 1 2 3 4 5 6 7)

## ---------------- log path ------------------
## four args: 1->gnn 2->adj 3->node 4->seed
create_log_file() {
    logFile=../logs/test/$dataset/$1-$2-$3-$4.log
    parentDir=$(dirname "$logFile")
    if [[ ! -d "$parentDir" ]]; then
        echo "didn't exist direction $parentDir and creating..."
        mkdir -p "$parentDir"
        echo "DONE"
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
echo "running for..."
# echo $file_paths_list

i=0
for adj in "${adjs[@]}"; do
    for gnn in "${gnns[@]}"; do
        for node in "${nodes[@]}"; do
            for seed in "${seeds[@]}"; do
                create_log_file $gnn $adj $node $seed
                echo "training and logging file $logFile"
                python -u test.py \
                    --log_path $logFile \
                    --conf_file ./config/new/$dataset/$gnn-$adj-$node.yml \
                    --save_path ./models \
                    --gpu ${gpus[$i]} \
                    --seed $seed 
                echo "DONE with PID $!"
            done
        done
    done
    ((i++))
done