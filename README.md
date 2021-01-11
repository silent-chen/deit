
# Train Deit

## Environment Setup
```buildoutcfg
pip install -r reuirements.txt
```

## Data preparation

The directory structure is the standard layout as following.

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```


## Original

To training the same model as described in DeiT. You should use the following comman and specify the model.  
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_small_patch16_224 --batch-size 128 --data-path /path/to/imagenet
```

## Distill

To distill the model use a teacher model in a hard distillation way, you should specify the teach model and whether use the distill token or the normal way. An example is given like:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_small_patch16_224 --batch-size 128 --data-path /path/to/imagenet --teacher_model /path/to/teacher_model_checkpoint [--distill_token]
```

## Fine-tune

To fine-tune a trained model on a higher resolution, you should specify the argument `--input-size` and `--resume`.
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env finetune.py --model deit_small_patch16_224 --batch-size 128 --input-size 384 --data-path /path/to/imagenet --resume /path/to/trained_model
```

## Relative-position

To use the 2D relative position, you should add the argument `--relative_position` to the training command. For example, to train deit small with relative position embedding, you should run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_small_patch16_224 --batch-size 128 --data-path /path/to/imagenet --relative_position
```
## itp script

The itp script is in the `itp` directory. To submit the job to cluster, you should specify the `--cluster` and the numbers of gpu you want to use `--gpus`. Here is an example:
```
python azureml_script.py --name ours_small_batch_size_128_8_gpus_lr_1e-3_warmup_epochs_20 --cluster itplabrr1cl1 --gpus 8
```

## lr_scheduler

Implemented the polynomial lr scheduler and integrated it with timm LRscheduler class. To use the polynomial LR scheduler, please specify the argument `--sched polynomial` 