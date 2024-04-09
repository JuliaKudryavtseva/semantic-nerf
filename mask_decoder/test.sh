#!/bin/bash


if [ $1 = "teatime" ]
then
    TEXT_PROMPTS=("apple" "brown teddy bear" "mug" "plate")
elif [ $1 = "waldo_kitchen" ]
then
  echo $1
else
  echo $1
  echo $1
fi


for prompt in "${TEXT_PROMPTS[@]}"; do

     python3 decode_masks.py --data-path $1 --text-prompt $prompt
     python3 decode_masks.py --data-path $1 --text-prompt $prompt --reg True
done