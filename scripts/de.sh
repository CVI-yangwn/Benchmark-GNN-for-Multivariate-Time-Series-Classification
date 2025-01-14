source activate gnn
cd /data/yangwennuo/code/MTSC/MTSC-Graph-benchmarking

pids=()  # store child pids of python to kill them all
trap 'echo "Caught signal; killing subprocesses..."; for pid in "${pids[@]}"; do echo ${pids}; kill ${pid} || true; done; echo "kill all and exit"; exit' SIGINT SIGTERM

##---------------- param --------------------
# ArticularyWordRecognition AtrialFibrillation BasicMotions
# Cricket
# "ArticularyWordRecognition" "AtrialFibrillation" "FingerMovements" "HandMovementDirection" "Heartbeat"
datasets=("Cricket")
# "mutual_information" "abspearson" "complete" "diffgraphlearn" "phase_locking_value"
adjs=("mutual_information" "abspearson" "complete" "diffgraphlearn")
# "raw" "differential_entropy" "power_spectral_density"
nodes=("differential_entropy")
# "chebnet" "gat" "gcn" "megat" "stgcn"
gnns=("chebnet" "gat" "gcn" "megat" "stgcn")
# 42 152 310
seeds=(42 152 310)
# 4 5 6 7 8 10 12 14 16 18 20
des=(4 5 6 7 8 10 12 14 16 18 20)
# 0 1 2 3 4 5 6 7
gpu=2

trainLog="../train"${gpu}".log"
## ---------------- log path ------------------
## four args: 1->gnn 2->adj 3->node 4->seed
create_log_file() {
    logFile=../logs/de/$dataset/$5/$1-$2-$3-$4.log
    parentDir=$(dirname "$logFile")
    if [[ ! -d "$parentDir" ]]; then
        echo "didn't exist direction $parentDir and creating..." >> train.log
        mkdir -p "$parentDir"
        echo "DONE" >> train.log
    fi
}

for dataset in "${datasets[@]}"; do
for de in "${des[@]}"; do
python ../change_yml_value.py $dataset $de
wait
for gnn in "${gnns[@]}"; do
    for adj in "${adjs[@]}"; do
        for node in "${nodes[@]}"; do
            for seed in "${seeds[@]}"; do
                time=$(date "+%Y%m%d-%H%M%S")
                create_log_file $gnn $adj $node $seed $de
                echo "training and logging file $logFile" >> train.log
                OMP_NUM_THREADS=8 python -u main.py \
                    --log_path $logFile \
                    --conf_file ./config/de/$dataset/$gnn-$adj-$node.yml \
                    --save_path ./models \
                    --gpu $gpu \
                    --seed $seed &
                waitPID=$!
                pids+=($waitPID)
                echo "wait "$gnn"-"$adj"-"$node"-"$seed"-"$de""
            done
        done
        wait $waitPID
    done
done
done
done