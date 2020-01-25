7th place solution to the TensorFlow 2.0 Question Answering competition

Solution summary: https://www.kaggle.com/c/tensorflow2-question-answering/discussion/127259

envorinment: python 3.6+, tensorflow 1.15

## Files
Most of the model code are based on [bert joint](https://github.com/google-research/language/tree/master/language/question_answering/bert_joint). Evaluation code are based on [official NQ metric](https://github.com/google-research-datasets/natural-questions), but modified for this competition.
- `prepare_nq_data.py`: pre-processing
- `jb_train_tpu.py`: training on TPU
- `jb_pred_tpu.py`: inference and evaluation of dev set on TPU
- `ensemble_and_tune.py`: tuning ensemble weights and thresholds
- `7th-place-submission.ipynb`: inference notebook, same as [this](https://www.kaggle.com/boliu0/7th-place-submission)
- `vocab_cased-nq.txt`: vocab file for cased model with special NQ tokens added
- `bert_config_cased.json`: config file for cased model

## scripts for the 3 single models
### model c: wwm, neg sampling, max_contexts=200, dev 64.5
```
# pre-processing (this step does not require TPU and could be distributed over multiple processes)
export do_lower_case=True
export max_contexts=200
export tfrecord_dir=fix_top_level_bug_max_contexts_200_0.01_0.04
for shard in {0..49} 
do 
	python3 prepare_nq_data.py --do_lower_case=$do_lower_case --tfrecord_dir=$tfrecord_dir --include_unknowns_answerable=0.01 --include_unknowns_unanswerable=0.04 --shard=$shard --max_contexts=$max_contexts
done

# training
export TPU_NAME=node-1
export train_batch_size=64
export learning_rate=4e-5
export model_suffix=_wwm_fix_top_level_bug_max_contexts_200_0.01_0.04
export train_precomputed_file=gs://<your_bucket>/tfrecords/fix_top_level_bug_max_contexts_200_0.01_0.04/nq-train.tfrecords-*
export init_checkpoint=gs://<your_bucket>/wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt
python3 jb_train_tpu.py --tpu=$TPU_NAME --model_suffix=${model_suffix} --train_batch_size=${train_batch_size} --learning_rate=${learning_rate} --train_precomputed_file=$train_precomputed_file --init_checkpoint=$init_checkpoint --num_train_epochs=1 

# evaluation (ckpt 9500 turned out to be the best)
export MODEL_SUFFIX=_wwm_fix_top_level_bug_max_contexts_200_0.01_0.04-64-4.00E-05
export CKPT_FROM=8000
export CKPT_TO=10000
export doc_stride=256
export do_lower_case=True
python3 jb_pred_tpu.py --tpu=$TPU_NAME --doc_stride=$doc_stride --model_suffix=$MODEL_SUFFIX --ckpt_from=$CKPT_FROM --ckpt_to=$CKPT_TO --eval_set=dev --do_predict=True --do_lower_case=$do_lower_case
```

### model d: wwm, neg sampling, stride=192, dev 63.8
```
# pre-processing (this step does not require TPU and could be distributed over multiple processes)
export do_lower_case=True
export doc_stride=192
export tfrecord_dir=stride_192_0.01_0.04
for shard in {0..49} 
do 
	python3 prepare_nq_data.py --do_lower_case=$do_lower_case --tfrecord_dir=$tfrecord_dir --include_unknowns_answerable=0.01 --include_unknowns_unanswerable=0.04 --shard=$shard --doc_stride=$doc_stride
done

# training
export TPU_NAME=node-1
export train_batch_size=64
export learning_rate=2e-5
export model_suffix=_wwm_stride_192_neg_0.01_0.04
export train_precomputed_file=gs://<your_bucket>/tfrecords/stride_192_0.01_0.04/nq-train.tfrecords-*
export init_checkpoint=gs://<your_bucket>/wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt
python3 jb_train_tpu.py --tpu=$TPU_NAME --model_suffix=${model_suffix} --train_batch_size=${train_batch_size} --learning_rate=${learning_rate} --train_precomputed_file=$train_precomputed_file --init_checkpoint=$init_checkpoint --num_train_epochs=1 

# evaluation (ckpt 7000 turned out to be the best)
export MODEL_SUFFIX=_wwm_stride_192_neg_0.01_0.04-64-2.00E-05
export CKPT_FROM=5000
export CKPT_TO=8000
export doc_stride=256
export do_lower_case=True
python3 jb_pred_tpu.py --tpu=$TPU_NAME --doc_stride=$doc_stride --model_suffix=$MODEL_SUFFIX --ckpt_from=$CKPT_FROM --ckpt_to=$CKPT_TO --eval_set=dev --do_predict=True --do_lower_case=$do_lower_case
```

### model e: wwm, neg sampling, cased, dev 63.3
```
# pre-processing (this step does not require TPU and could be distributed over multiple processes)
export do_lower_case=False
export tfrecord_dir=fix_top_level_bug_cased_0.01_0.04
for shard in {0..49} 
do 
	python3 prepare_nq_data.py --do_lower_case=$do_lower_case --tfrecord_dir=$tfrecord_dir --include_unknowns_answerable=0.01 --include_unknowns_unanswerable=0.04 --shard=$shard
done

# training
export TPU_NAME=node-1
export train_batch_size=64
export learning_rate=4.5e-5
export model_suffix=_wwm_cased_fix_top_level_bug_0.01_0.04
export train_precomputed_file=gs://<your_bucket>/tfrecords/fix_top_level_bug_cased_0.01_0.04/nq-train.tfrecords-*
export init_checkpoint=gs://<your_bucket>/wwm_cased_L-24_H-1024_A-16/bert_model.ckpt
export do_lower_case=False
python3 jb_train_tpu.py --tpu=$TPU_NAME --model_suffix=${model_suffix} --train_batch_size=${train_batch_size} --learning_rate=${learning_rate} --train_precomputed_file=$train_precomputed_file --init_checkpoint=$init_checkpoint --num_train_epochs=1 --do_lower_case=${do_lower_case}

# evaluation (ckpt 8500 turned out to be the best)
export MODEL_SUFFIX=_wwm_cased_fix_top_level_bug_0.01_0.04-64-4.50E-05
export CKPT_FROM=6000
export CKPT_TO=8500
export doc_stride=256
export do_lower_case=False
python3 jb_pred_tpu.py --tpu=$TPU_NAME --doc_stride=$doc_stride --model_suffix=$MODEL_SUFFIX --ckpt_from=$CKPT_FROM --ckpt_to=$CKPT_TO --eval_set=dev --do_predict=True --do_lower_case=$do_lower_case
```




