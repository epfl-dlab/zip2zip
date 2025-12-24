if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
  echo "Usage: $0 <config_file> <seed> <rid>"
  exit 1
fi

CONFIG_FILE=$1
SEED=$2
RID=$3

export PROCESSES_PER_NODE=$RUNAI_NUM_OF_GPUS

echo "Num GPU: $PROCESSES_PER_NODE"
echo "Config file: $CONFIG_FILE"
echo "Seed: $SEED"
echo "Run ID: $RID"

# THIS PART IS THE SAME FOR ALL THE SCRIPTS
###########################################
source /opt/conda/bin/activate zip2zip

cd ozz
# git pull

###########################################

torchrun --standalone --nproc-per-node="$PROCESSES_PER_NODE" -m train --config="$CONFIG_FILE" --seed "$SEED" --rid "$RID"
