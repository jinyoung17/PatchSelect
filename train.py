import os
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from model.main_model import RGN
from utils.util import EditingJsonDataset, EditingSingleImageDataset
from utils.util2 import compose_text_with_templates, get_augmentations_template
import utils.misc as misc
from torchvision.utils import save_image


# Configure optimizer with separate learning rates 
# for patch scoring MLP, fusion weights, and text projection
def configure_optimizers(model, lr, betas=(0.9, 0.96), weight_decay=1e-2):
    mlp_params = list(model.module.score_predict.score_predict.parameters())
    scalar_ws = [model.module.score_predict.weights]
    tx_params = list(model.module.text_proj.parameters())

    return torch.optim.AdamW(
        [
            {"params": mlp_params, "lr": lr * 0.2, "weight_decay": weight_decay},
            {"params": scalar_ws,  "lr": lr * 0.6, "weight_decay": 0.0},
            {"params": tx_params, "lr": lr,       "weight_decay": weight_decay},
        ],
        betas=betas,
        eps=1e-8
    )


def train(args, lr_schedule, model, template, data_loader, optimizer, device_id):
    rank = dist.get_rank()

    for epoch in range(1, args.epochs + 1):
        data_loader.sampler.set_epoch(epoch)

        for step, (imgs, o_prompt, e_prompt) in enumerate(tqdm(data_loader, disable=(rank != 0))):
            imgs = imgs.to(device_id, non_blocking=True)
            src_prompt = o_prompt[0]
            tgt_prompt = compose_text_with_templates(e_prompt[0], template)


            # Extract CLIP text embeddings for the target prompt
            text_emb = model.module.clip_extractor.get_text_embedding(tgt_prompt, device_id)

            # Generate patch-level editing mask
            mask = model.module.get_patch_mask(imgs, text_emb)

            # Invert mask for diffusion-based blending
            # (token_sparse_ratio=0.6 corresponds to 40% editable regions)
            mask = 1.0 - mask.unsqueeze(1)

            # Upsample patch mask to image resolution
            mask = F.interpolate(mask, size=(args.image_size, args.image_size), mode="nearest")

            results, x_edit = model.module.generate_result(
                imgs, mask, tgt_prompt, ste_blend=True
            )

            loss, *_ = model.module.get_loss(
                imgs, results, tgt_prompt, src_prompt
            )

            loss.backward()

            # Update model parameters with gradient accumulation 
            if (step + 1) % args.accum_grad == 0:
                optimizer.step()
                lr_schedule.step()
                optimizer.zero_grad()

            if rank == 0 and args.output_dir is not None:
                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir, exist_ok=True)

                save_path = os.path.join(
                    args.output_dir,
                    f"epoch{epoch}_xedit.png"
                )
  
                save_image(x_edit[0], save_path)

        if rank == 0 and epoch % args.ckpt_interval == 0:
            os.makedirs(args.save_path, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(args.save_path, f"epoch_{epoch}.pth")
            )

def main(args):
    dist.init_process_group("nccl", init_method="env://")
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()

    # Set random seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    template = get_augmentations_template()

    model = RGN(
        image_size=args.image_size,
        device=device_id,
        args=args
    ).to(device_id)

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device_id], find_unused_parameters=True
    )

    if args.json_file:
        train_dataset = EditingJsonDataset(args, args.per_image_iteration)
    else:
        train_dataset = EditingSingleImageDataset(args, args.per_image_iteration)

    sampler = DistributedSampler(train_dataset)
    data_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    optimizer = configure_optimizers(model, args.lr)
    total_steps = len(data_loader) * args.epochs
    lr_schedule = CosineAnnealingLR(optimizer, T_max=total_steps)

    train(
        args,
        lr_schedule,
        model,
        template,
        data_loader,
        optimizer,
        device_id
    )

    dist.destroy_process_group()


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--json_file', type=str, default=None)
    parser.add_argument('--per_image_iteration', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--save_path', type=str, default='./checkpoints')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--ckpt_interval', type=int, default=10)
    parser.add_argument('--accum_grad', type=int, default=1)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--token_sparse_ratio', type=float, default=0.6)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args_parser()
    main(args)
