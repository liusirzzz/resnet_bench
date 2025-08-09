#!/usr/bin/env python3
import argparse
import time
import torch
import torch.nn as nn
import torchvision.models as models

def get_model(name: str):
    name = name.lower()
    if name == "resnet18":
        return models.resnet18(weights=None)
    if name == "resnet34":
        return models.resnet34(weights=None)
    if name == "resnet50":
        return models.resnet50(weights=None)
    if name == "resnet101":
        return models.resnet101(weights=None)
    if name == "resnet152":
        return models.resnet152(weights=None)
    if name == "resnext50_32x4d":
        return models.resnext50_32x4d(weights=None)
    if name == "resnext101_32x8d":
        return models.resnext101_32x8d(weights=None)
    raise ValueError(f"Unknown model: {name}")

def main():
    parser = argparse.ArgumentParser(description="ResNet throughput benchmark")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--precision", type=str, default="float16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--iters", type=int, default=5, help="timed iterations")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--train", action="store_true", help="enable training benchmark")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--compile", action="store_true", help="torch.compile model")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is required"
    torch.cuda.set_device(args.device)

    # Speed-oriented backend settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # Seed (optional)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    device = torch.device("cuda", args.device)

    model = get_model(args.model)
    model = model.to(device)
    model = model.to(memory_format=torch.channels_last)

    # Precision handling
    prec = args.precision
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[prec]

    # Inference path: cast model & inputs for pure-precision throughput
    use_amp_train = False
    scaler = None
    if not args.train:
        if prec == "float16":
            model = model.half()
        elif prec == "bfloat16":
            model = model.bfloat16()
        # float32: keep as is
        amp_context = torch.no_grad()  # not used, we rely on inference_mode below
    else:
        # Training path: AMP for stability/perf, model stays fp32
        if prec in ("float16", "bfloat16"):
            use_amp_train = True
            if prec == "float16":
                scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            use_amp_train = False

    # Optional torch.compile (can help on some GPUs/PyTorch versions; costs warmup time)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode="max-autotune")

    model.eval() if not args.train else model.train()

    batch = args.batch_size
    C, H, W = 3, args.image_size, args.image_size

    # Create synthetic data directly on GPU to avoid PCIe overhead
    x = torch.randn(batch, C, H, W, device=device, dtype=dtype)
    x = x.to(memory_format=torch.channels_last)
    labels = torch.randint(0, 1000, (batch,), device=device, dtype=torch.long)

    criterion = nn.CrossEntropyLoss().to(device)
    # Optimizer (training only)
    optimizer = None
    if args.train:
        # Fused/foreach optimizer if available for lower overhead
        fused_ok = "fused" in nn.modules.__dict__ or "fused" in torch.optim.SGD.__init__.__code__.co_varnames
        foreach_ok = True
        try:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4,
                foreach=foreach_ok, fused=True  # fused may not be supported on all versions
            )
        except TypeError:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4,
                foreach=foreach_ok
            )

    # Print args in the requested style
    print(argparse.Namespace(device=args.device, model=args.model, precision=prec, train=args.train))

    # Warmup
    if not args.train:
        with torch.inference_mode():
            for _ in range(args.warmup):
                _ = model(x)
    else:
        for _ in range(args.warmup):
            optimizer.zero_grad(set_to_none=True)
            if use_amp_train:
                with torch.cuda.amp.autocast(dtype=dtype):
                    out = model(x)
                    loss = criterion(out, labels)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                out = model(x)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

    # Timed iterations (CUDA events for accurate GPU time)
    for i in range(args.iters):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        if not args.train:
            with torch.inference_mode():
                _ = model(x)
        else:
            optimizer.zero_grad(set_to_none=True)
            if use_amp_train:
                with torch.cuda.amp.autocast(dtype=dtype):
                    out = model(x)
                    loss = criterion(out, labels)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                out = model(x)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        sec = ms / 1000.0
        ips = batch / sec
        print(f"Iteration {i}, {ips:.2f} images/s in {sec:.3f}s.")

if __name__ == "__main__":
    main()