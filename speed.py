from timm.models import create_model
from timm.models.vision_transformer import VisionTransformer
import torch
import torch.nn as nn
import time
import argparse
import models
from ptflops import get_model_complexity_info
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
def main():
    parser = argparse.ArgumentParser('Training speed test', add_help=False)
    parser.add_argument('--model', default='deit_base_patch16_224', type=str)
    parser.add_argument('--num_classes', default=1000, type=int)
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--warm-up', type=int, default=5)
    parser.add_argument('--total-runs', type=int, default=35)
    parser.add_argument('--image-size', type=int, default=224)

    args = parser.parse_args()
    device = 'cuda:0'
    torch.cuda.set_device(device)
    batch_size = args.batch_size
    # model = VisionTransformer(
    #     patch_size=16, embed_dim=256, depth=8, num_heads=4, mlp_ratio=4, qkv_bias=True,
    #     norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=args.drop, drop_path_rate=args.drop_path,)
    # model.default_cfg = _cfg()

    model = create_model(model_name=args.model,
                         num_classes=args.num_classes,
                         pretrained=False,
                         drop_rate=args.drop,
                         drop_path_rate=args.drop_path,
                         drop_block_rate=args.drop_block,
                         )
    model = model.cuda()
    inputs = torch.randn(batch_size, 3, args.image_size, args.image_size).cuda()
    macs, params = get_model_complexity_info(model, (3, args.image_size, args.image_size), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    runs_warm_up = args.warm_up
    final_runs = args.total_runs
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for run in range(final_runs):
            outputs = model(inputs)
            if run == runs_warm_up:
                start_time = time.time()
    time_eclapse = time.time() - start_time
    print("Avarage images/s for 30 runs is : {} images/s , total time: {}, total images: {}".format((final_runs-runs_warm_up)*batch_size/time_eclapse, time_eclapse, (final_runs-runs_warm_up)*batch_size))

if __name__ == "__main__":
    main()