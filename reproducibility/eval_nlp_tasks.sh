# Phi-3.5-4B

NUM_SAMPLES=500

python eval.py  --adapter=checkpoints/evqn/model_7000.safetensors --tasks ai2_arc,openbookqa,piqa,winogrande,commonsense_qa,lambada,mathqa,hellaswag  --limit $NUM_SAMPLES
python eval.py  --adapter=checkpoints/HCPt/model_7000.safetensors --tasks ai2_arc,openbookqa,piqa,winogrande,commonsense_qa,lambada,mathqa,hellaswag  --limit $NUM_SAMPLES
lm_eval --model hf \
    --model_args pretrained=microsoft/Phi-3.5-mini-instruct,max_length=$CTX,dtype=auto \
    --tasks ai2_arc,openbookqa,piqa,winogrande,commonsense_qa,lambada,mathqa,hellaswag   \
    --device cuda:0 \
    --batch_size auto --limit $NUM_SAMPLES --trust_remote_code --confirm_run_unsafe_code


# Phi-3-14B

python eval.py  --adapter=checkpoints/QQsH/model_7000.safetensors --tasks ai2_arc,openbookqa,piqa,winogrande,commonsense_qa,lambada,mathqa,hellaswag  --limit $NUM_SAMPLES

python eval.py  --adapter=checkpoints/bTgh/model_7000.safetensors --tasks ai2_arc,openbookqa,piqa,winogrande,commonsense_qa,lambada,mathqa,hellaswag --limit $NUM_SAMPLES

lm_eval --model hf \
    --model_args pretrained=microsoft/Phi-3-medium-4k-instruct,max_length=$CTX,dtype=auto \
    --tasks ai2_arc,openbookqa,piqa,winogrande,commonsense_qa,lambada,mathqa,hellaswag   \
    --device cuda:0 \
    --batch_size auto --limit $NUM_SAMPLES --trust_remote_code --confirm_run_unsafe_code
