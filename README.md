
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
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_small_patch16_224 --batch-size 128 --data-path /path/to/imagenet --teacher_model /path/to/teacher_model_checkpoint [--distill token]
```

## Fine-tune

To fine-tune a trained model on a higher resolution, you should specify the argument `--input-size` and `--resume`.
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env finetune.py --model deit_small_patch16_224 --batch-size 128 --input-size 384 --data-path /path/to/imagenet --resume /path/to/trained_model
```

## itp script

The itp script is in the `itp` directory.
