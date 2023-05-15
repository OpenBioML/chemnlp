#!/bin/bash
hf_subdirs=( accelerate datasets hub )
for sub_dir in "${hf_subdirs[@]}"
do
   cp -R ~/.cache/huggingface/$sub_dir/* /fsx/proj-chemnlp/hf_cache/$sub_dir
   rm -rf ~/.cache/huggingface/$sub_dir
done
