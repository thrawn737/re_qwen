import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

revisit_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(revisit_path, 'src'))

from utils import *

from transformers import AutoModel, AutoTokenizer
from internvl_utils import load_image_for_internvl, evolve_revisit_generate
from chair_loader import CHAIRDataset

import warnings
warnings.filterwarnings(action='ignore')

import aiofiles
import asyncio

def parse_args():
    parser = argparse.ArgumentParser(description="CHAIR on InternVL")
    parser.add_argument("--model_path", type=str, required=True, help="model")
    parser.add_argument("--model_base", type=str, default="internvl")

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=1)

    parser.add_argument("--data_path", type=str, default="/root/data/coco/val2014/", help="data path")
    parser.add_argument("--anno_path", type=str, default="/root/data/coco/annotations/instances_val2014.json")
    parser.add_argument("--output_path", type=str, default=None, help="output path")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="num workers")

    parser.add_argument("--num_eval_samples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--exp_name", type=str, default='000')

    parser.add_argument("--do_sample", type=str2bool, default=False)
    parser.add_argument("--use_revisit", type=str2bool, default=True)
    parser.add_argument("--early_exit_layers", type=str, default="last", choices=["last", "all"], help="early exit layers")
    parser.add_argument("--relative_top", type=float, default=1e-6, help="relative top")

    args = parser.parse_known_args()[0]
    return args

async def main():
    args = parse_args()
    print("args: ", args)
    setup_dist(args.seed, args.batch_size)
    if args.output_path is not None:
        output_path = args.output_path
    else:
        output_path = os.path.join(revisit_path, "output", "InternVL3")
    output_file = os.path.join(output_path, f"{args.exp_name}.jsonl")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print("output_file: ", output_file)

    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval().cuda()

    evolve_revisit_generate(model)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    print("Model loaded")

    chair_dataset = CHAIRDataset(
        data_path=args.data_path,
        anno_path=args.anno_path,
        trans=None,
        model=args.model_base
    )
    chair_loader = DataLoader(
        chair_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        drop_last=False
    )

    qs =  "Please describe this image in detail."

    generation_config = dict(
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        use_revisit=args.use_revisit,
        early_exit_layers=args.early_exit_layers,
        relative_top=args.relative_top,
    )

    for batch_id, data in tqdm(enumerate(chair_loader), total=args.num_eval_samples):

        # early stop for debuggging purpose
        # if batch_id == 20:
            # break

        if batch_id == args.num_eval_samples:
            break
            
        image_id = data['image_id']
        image_path = data["image_path"]

        pixel_values = load_image_for_internvl(image_path[0], is_str=True).cuda().to(torch.bfloat16)

        
        with torch.inference_mode():
            with torch.no_grad():
                outputs = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    question=qs,
                    generation_config=generation_config,
                    verbose=False,
                )

        print(f"[VQA for {args.exp_name}]")
        print(f"V: {image_path[0]}")
        print(f"Q: {qs}")
        print(f"A: {outputs}")
        print(f"="*50)

        img_save = {}
        img_save["image_id"] = image_id.item()
        img_save["caption"] = outputs

        # dump metric file
        async with aiofiles.open(output_file, "a") as f:
            await f.write(json.dumps(img_save) + '\n')
    
    print(f"Results saved to {output_file}")
    terminate_dist()

if __name__ == "__main__":
    asyncio.run(main())