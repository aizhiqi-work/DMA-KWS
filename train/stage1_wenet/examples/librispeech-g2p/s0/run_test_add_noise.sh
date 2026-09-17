#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1;

# Automatically detect number of gpus
if command -v nvidia-smi &> /dev/null; then
  num_gpus=$(nvidia-smi -L | wc -l)
  gpu_list=$(seq -s, 4 5)
else
  num_gpus=-1
  gpu_list="-1"
fi
# You can also manually specify CUDA_VISIBLE_DEVICES
# if you don't want to utilize all available GPU resources.
export CUDA_VISIBLE_DEVICES="${gpu_list}"
echo "CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"

stage=0 # start from 0 if you need to start from data preparation
stop_stage=5

# You should change the following two parameters for multiple machine training,
# see https://pytorch.org/docs/stable/elastic/run.html
HOST_NODE_ADDR="localhost:0"
num_nodes=1
job_id=2023

nj=16
dict=data/dict/lang_char.txt

# data_type can be `raw` or `shard`. Typically, raw is used for small dataset,
# `shard` is used for large dataset which is over 1k hours, and `shard` is
# faster on reading data and training.
data_type=raw
num_utts_per_shard=1000

train_set=train
train_config=conf/train_conformer_addnoise.yaml
dir=exp/conformer-addnoise
tensorboard_dir=tensorboard/conformer-addnoise
checkpoint=
num_workers=8
prefetch=10

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
average_num=10
# decode_modes="ctc_prefix_beam_search"
decode_modes="ctc_greedy_search"

train_engine=torch_ddp

deepspeed_config=conf/ds_stage2.json
deepspeed_save_states="model_only"

. tools/parse_options.sh || exit 1;

# if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
#   echo "stage -1: Data Download"
#   local/download_and_untar.sh ${data} ${data_url} data_aishell
#   local/download_and_untar.sh ${data} ${data_url} resource_aishell
# fi

# if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
#   # Data preparation
#   local/aishell_data_prep.sh ${data}/data_aishell/wav \
#     ${data}/data_aishell/transcript
# fi


# if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
#   # remove the space between the text labels for Mandarin dataset
#   for x in train dev test; do
#     cp data/${x}/text data/${x}/text.org
#     paste -d " " <(cut -f 1 -d" " data/${x}/text.org) \
#       <(cut -f 2- -d" " data/${x}/text.org | tr -d " ") \
#       > data/${x}/text
#     rm data/${x}/text.org
#   done

#   tools/compute_cmvn_stats.py --num_workers 16 --train_config $train_config \
#     --in_scp data/${train_set}/wav.scp \
#     --out_cmvn data/$train_set/global_cmvn
# fi

# if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
#   echo "Make a dictionary"
#   mkdir -p $(dirname $dict)
#   echo "<blank> 0" > ${dict}  # 0 is for "blank" in CTC
#   echo "<unk> 1"  >> ${dict}  # <unk> must be 1
#   echo "<sos/eos> 2" >> $dict
#   tools/text2token.py -s 1 -n 1 data/train/text | cut -f 2- -d" " \
#     | tr " " "\n" | sort | uniq | grep -a -v -e '^\s*$' | \
#     awk '{print $0 " " NR+2}' >> ${dict}
# fi

# if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
#   echo "Prepare data, prepare required format"
#   for x in dev test ${train_set}; do
#     if [ $data_type == "shard" ]; then
#       tools/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
#         --num_threads 16 data/$x/wav.scp data/$x/text \
#         $(realpath data/$x/shards) data/$x/data.list
#     else
#       tools/make_raw_list.py data/$x/wav.scp data/$x/text \
#         data/$x/data.list
#     fi
#   done
# fi

# if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
#   mkdir -p $dir
#   num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
#   # Use "nccl" if it works, otherwise use "gloo"
#   # NOTE(xcsong): deepspeed fails with gloo, see
#   #   https://github.com/microsoft/DeepSpeed/issues/2818
#   dist_backend="nccl"

#   # train.py rewrite $train_config to $dir/train.yaml with model input
#   # and output dimension, and $dir/train.yaml will be used for inference
#   # and export.
#   if [ ${train_engine} == "deepspeed" ]; then
#     echo "$0: using deepspeed"
#   else
#     echo "$0: using torch ddp"
#   fi

#   # NOTE(xcsong): Both ddp & deepspeed can be launched by torchrun
#   # NOTE(xcsong): To unify single-node & multi-node training, we add
#   #               all related args. You should change `nnodes` &
#   #               `rdzv_endpoint` for multi-node, see
#   #               https://pytorch.org/docs/stable/elastic/run.html#usage
#   #               https://github.com/wenet-e2e/wenet/pull/2055#issuecomment-1766055406
#   #               `rdzv_id` - A user-defined id that uniquely identifies the worker group for a job.
#   #                           This id is used by each node to join as a member of a particular worker group.
#   #               `rdzv_endpoint` - The rendezvous backend endpoint; usually in form <host>:<port>.
#   # NOTE(xcsong): In multi-node training, some clusters require special NCCL variables to set prior to training.
#   #               For example: `NCCL_IB_DISABLE=1` + `NCCL_SOCKET_IFNAME=enp` + `NCCL_DEBUG=INFO`
#   #               without NCCL_IB_DISABLE=1
#   #                   RuntimeError: NCCL error in: ../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1269, internal error, NCCL Version xxx
#   #               without NCCL_SOCKET_IFNAME=enp  (IFNAME could be get by `ifconfig`)
#   #                   RuntimeError: The server socket has failed to listen on any local network address. The server socket has failed to bind to [::]:xxx
#   #               ref: https://github.com/google/jax/issues/13559#issuecomment-1343573764
#   echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
#   torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
#            --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
#     wenet/bin/train.py \
#       --train_engine ${train_engine} \
#       --config $train_config \
#       --data_type  $data_type \
#       --train_data data/train.list \
#       --cv_data data/dev.list \
#       ${checkpoint:+--checkpoint $checkpoint} \
#       --model_dir $dir \
#       --tensorboard_dir ${tensorboard_dir} \
#       --ddp.dist_backend $dist_backend \
#       --num_workers ${num_workers} \
#       --prefetch ${prefetch} \
#       --pin_memory \
#       --deepspeed_config ${deepspeed_config} \
#       --deepspeed.save_states ${deepspeed_save_states}
# fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # Test model, please specify the model you want to test by --checkpoint
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg_${average_num}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path $dir  \
      --num ${average_num} \
      --val_best
  fi
  # Please specify decoding_chunk_size for unified streaming and
  # non-streaming model. The default value is -1, which is full chunk
  # for non-streaming inference.
  decoding_chunk_size=
  ctc_weight=1
  reverse_weight=0
  python wenet/bin/recognize.py --gpu 0 \
    --modes $decode_modes \
    --config $dir/train.yaml \
    --data_type $data_type \
    --test_data data/lists/test-clean.list \
    --checkpoint $decode_checkpoint \
    --beam_size 10 \
    --batch_size 64 \
    --blank_penalty 0.0 \
    --ctc_weight $ctc_weight \
    --reverse_weight $reverse_weight \
    --result_dir $dir/test-clean \
    ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
  for mode in ${decode_modes}; do
    python tools/compute-wer.py --char=1 --v=1 \
      data/lists/test-clean.txt $dir/test-clean/$mode/text > $dir/test-clean/$mode/wer
  done

  python wenet/bin/recognize.py --gpu 0 \
    --modes $decode_modes \
    --config $dir/train.yaml \
    --data_type $data_type \
    --test_data data/lists/test-other.list \
    --checkpoint $decode_checkpoint \
    --beam_size 10 \
    --batch_size 64 \
    --blank_penalty 0.0 \
    --ctc_weight $ctc_weight \
    --reverse_weight $reverse_weight \
    --result_dir $dir/test-other \
    ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
  for mode in ${decode_modes}; do
    python tools/compute-wer.py --char=1 --v=1 \
      data/lists/test-other.txt $dir/test-other/$mode/text > $dir/test-other/$mode/wer
  done
fi