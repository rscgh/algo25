#!/bin/bash

#srun --nodes=1 --ntasks-per-node=1 --cpus-per-task 2 --mem=40gb --time=10:00:00 --partition scc-cpu --pty bash -i
#srun --nodes=1 --ntasks-per-node=1 --cpus-per-task 2 --mem=60gb --time=10:00:00 --partition scc-cpu --pty bash -i

#model_ids=("Llama-3.1-8B:4L7T1000W" "whisper-small:4L18T29S" "algo_slowr50:None" "SmolLM2-1.7B:4L3T1000W" "dinov2.4LCLS" "whisper-large-v3:4L18T29S" "Qwen2.5-7B:4L3T1000W")
model_ids=("llama-3-8b-bnb-4bit-contpretr-lora-friends-0.1:4L7T1000W")
stim_windows=(1 2 3 4)
n_feats=(100 500 1000)
#n_feats=(1000)

for id in "${model_ids[@]}"; do
  for sw in "${stim_windows[@]}"; do
    for n_feat in "${n_feats[@]}"; do
      python -u regress_algo25.py --embedding_comb "${id}:${sw}:${n_feat}"
    done
  done
done

#model_ids=("Llama-3.1-8B:4L7T1000W" "whisper-small:4L18T29S" "algo_slowr50:None" "SmolLM2-1.7B:4L3T1000W" "whisper-large-v3:4L18T29S")
model_ids=() #("whisper-large-v3:4L18T29S" "Qwen2.5-7B:4L3T1000W")
stim_windows=(1 3 4)
n_feats=(1500 2000)

for id in "${model_ids[@]}"; do
  for sw in "${stim_windows[@]}"; do
    for n_feat in "${n_feats[@]}"; do
      python -u regress_algo25.py --embedding_comb "${id}:${sw}:${n_feat}"
    done
  done
done


#id="llama-3-8b-bnb-4bit-contpretr-lora-friends-0.1:4L7T1000W"
#sw=3
#n_feat=2000
#python -u regress_algo25.py --embedding_comb "${id}:${sw}:${n_feat}"