import argparse
import os
import csv
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm


# ── Metrics ──────────────────────────────────────────────────────────────────

def computeIOU(output, target):
    output = torch.argmax(output, dim=1).flatten()
    target = target.flatten()
    no_ignore = target.ne(255).cuda()
    output = output.masked_select(no_ignore)
    target = target.masked_select(no_ignore)
    intersection = torch.sum(output * target)
    union = torch.sum(target) + torch.sum(output) - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou if iou == iou else torch.tensor(0.0).float().cuda()


def computeAccuracy(output, target):
    output = torch.argmax(output, dim=1).flatten()
    target = target.flatten()
    no_ignore = target.ne(255).cuda()
    output = output.masked_select(no_ignore)
    target = target.masked_select(no_ignore)
    correct = torch.sum(output.eq(target))
    return correct.float() / len(target)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model():
    net = models.segmentation.fcn_resnet50(
        pretrained=False, num_classes=2, pretrained_backbone=False
    )
    net.backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def convertBNtoGN(module, num_groups=16):
        if isinstance(module, nn.modules.batchnorm.BatchNorm2d):
            mod = nn.GroupNorm(num_groups, module.num_features, eps=module.eps, affine=module.affine)
            if module.affine:
                mod.weight.data = module.weight.data.clone().detach()
                mod.bias.data = module.bias.data.clone().detach()
            return mod
        for name, child in module.named_children():
            module.add_module(name, convertBNtoGN(child, num_groups=num_groups))
        return module

    return convertBNtoGN(net)


# ── Data loading ──────────────────────────────────────────────────────────────

def getArrFlood(fname):
    return rasterio.open(fname).read()


def download_flood_water_data_from_list(file_list, root=None):
    data = []
    for im_fname, mask_fname in file_list:
        if root:
            im_fname = os.path.join(root, im_fname)
            mask_fname = os.path.join(root, mask_fname)
        if not os.path.exists(im_fname) or not os.path.exists(mask_fname):
            continue
        arr_x = np.nan_to_num(getArrFlood(im_fname))
        arr_y = getArrFlood(mask_fname)
        arr_y[arr_y == -1] = 255
        arr_x = np.clip(arr_x, -50, 1)
        arr_x = (arr_x + 50) / 51
        data.append((arr_x, arr_y))
    return data


def load_data_from_csv(csv_path, input_root, label_root):
    files = []
    with open(csv_path) as f:
        for line in csv.reader(f):
            files.append((os.path.join(input_root, line[0]), os.path.join(label_root, line[1])))
    return download_flood_water_data_from_list(files)


# ── Preprocessing (test) ──────────────────────────────────────────────────────

norm = transforms.Normalize([0.6851, 0.5235], [0.0820, 0.1102])


def process_test_im(data):
    x, y = data
    im, label = x.copy(), y.copy()
    im_c1 = Image.fromarray(im[0]).resize((512, 512))
    im_c2 = Image.fromarray(im[1]).resize((512, 512))
    label = Image.fromarray(label.squeeze()).resize((512, 512))

    crops = [(0, 0, 256, 256), (0, 256, 256, 256), (256, 0, 256, 256), (256, 256, 256, 256)]
    im_c1s = [F.crop(im_c1, *c) for c in crops]
    im_c2s = [F.crop(im_c2, *c) for c in crops]
    labels = [F.crop(label, *c) for c in crops]

    ims = torch.stack([
        torch.stack([transforms.ToTensor()(c1).squeeze(), transforms.ToTensor()(c2).squeeze()])
        for c1, c2 in zip(im_c1s, im_c2s)
    ])
    ims = norm(ims)

    labels = torch.stack([transforms.ToTensor()(lbl).squeeze() for lbl in labels])
    if torch.sum(labels.gt(.003) * labels.lt(.004)):
        labels *= 255
    labels = labels.round()

    return ims, labels


# ── Test loop ─────────────────────────────────────────────────────────────────

def test_loop(test_loader, net):
    net = net.eval().cuda()
    count = 0
    iou = 0.0
    accuracy = 0.0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            outputs = net(images.cuda())
            iou += computeIOU(outputs["out"], labels.cuda())
            accuracy += computeAccuracy(outputs["out"], labels.cuda())
            count += 1

    print(f"\nTest IoU:       {iou / count:.6f}")
    print(f"Test Accuracy:  {accuracy / count:.6f}")


# ── Inspect (original) ────────────────────────────────────────────────────────

def inspect_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    print(f"{'='*60}")
    print(f"Checkpoint: {path}")
    print(f"{'='*60}")

    if isinstance(ckpt, dict):
        print(f"\nType: state_dict (dict)")
        print(f"Total entries: {len(ckpt)}")
        print(f"\n{'Key':<55} {'Shape':<20} {'Dtype':<10} {'Size':<10}")
        print(f"{'-'*95}")
        total_params = 0
        for key, val in ckpt.items():
            if hasattr(val, "shape"):
                size = val.numel() if hasattr(val, "numel") else np.prod(val.shape)
                total_params += size
                print(f"{key:<55} {str(list(val.shape)):<20} {str(val.dtype):<10} {size:<10}")
            else:
                print(f"{key:<55} {str(type(val).__name__):<20}")

        print(f"{'-'*95}")
        print(f"{'Total parameters:':<55} {total_params:<20}")
        print(f"{'Total size (MB):':<55} {total_params * 4 / 1024 / 1024:<20.2f}")

        has_nan = any(torch.isnan(val).any().item() for val in ckpt.values() if hasattr(val, "isnan"))
        has_inf = any(torch.isinf(val).any().item() for val in ckpt.values() if hasattr(val, "isinf"))
        if has_nan:
            print(f"{'WARNING: Contains NaN values':<55}")
        if has_inf:
            print(f"{'WARNING: Contains Inf values':<55}")

        return ckpt

    elif hasattr(ckpt, "state_dict"):
        sd = ckpt.state_dict()
        print(f"\nType: {type(ckpt).__name__} (full model)")
        print(f"Architecture: {ckpt.__class__.__name__}")
        print(f"State dict entries: {len(sd)}")
        return sd
    else:
        print(f"\nType: {type(ckpt).__name__}")
        print(f"Content: {str(ckpt)[:500]}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate_checkpoint(checkpoint_path, test_loader):
    print(f"\n{'='*60}")
    print(f"Evaluating: {checkpoint_path}")
    print(f"{'='*60}")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        print("  Skipped — not a state_dict")
        return
    net = build_model()
    net.load_state_dict(state_dict, strict=False)
    test_loop(test_loader, net)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect / test a PyTorch checkpoint")
    parser.add_argument("path", nargs="?", help="Path to checkpoint file (.cp, .pt, .pth)")
    parser.add_argument("--test", action="store_true", help="Run evaluation on test set")
    parser.add_argument("--all", action="store_true", help="Evaluate all checkpoints in a directory")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory with checkpoint files")
    parser.add_argument("--data-dir", default="files", help="Root data directory")
    parser.add_argument("--splits-dir", default="splits", help="Directory with CSV split files")
    parser.add_argument("--batch-size", type=int, default=4, help="Test batch size")
    args = parser.parse_args()

    if args.all:
        import glob
        paths = sorted(glob.glob(os.path.join(args.checkpoint_dir, "*.cp")))
        if not paths:
            print(f"No .cp files found in {args.checkpoint_dir}")
            exit(1)
        print(f"Found {len(paths)} checkpoints in {args.checkpoint_dir}")

        print("Loading test data once...")
        test_csv = os.path.join(args.splits_dir, "flood_test_data.csv")
        if not os.path.exists(test_csv):
            print(f"ERROR: test split not found at {test_csv}")
            exit(1)
        test_data = load_data_from_csv(test_csv,
            os.path.join(args.data_dir, "S1"), os.path.join(args.data_dir, "Labels"))
        print(f"Loaded {len(test_data)} test samples.")

        class TestDataset(torch.utils.data.Dataset):
            def __init__(self, data_list):
                self.data_list = data_list
            def __getitem__(self, i):
                return process_test_im(self.data_list[i])
            def __len__(self):
                return len(self.data_list)

        test_dataset = TestDataset(test_data)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda x: (torch.cat([a[0] for a in x], 0), torch.cat([a[1] for a in x], 0))
        )

        for path in paths:
            evaluate_checkpoint(path, test_loader)

    elif args.path:
        state_dict = inspect_checkpoint(args.path)

        if args.test and state_dict is not None:
            print(f"\n{'='*60}")
            print("Building model and loading weights...")
            net = build_model()
            net.load_state_dict(state_dict, strict=False)
            print("Model loaded.")

            print("Loading test data...")
            test_csv = os.path.join(args.splits_dir, "flood_test_data.csv")
            if not os.path.exists(test_csv):
                print(f"ERROR: test split not found at {test_csv}")
                exit(1)
            test_data = load_data_from_csv(test_csv,
                os.path.join(args.data_dir, "S1"), os.path.join(args.data_dir, "Labels"))
            print(f"Loaded {len(test_data)} test samples.")

            class TestDataset(torch.utils.data.Dataset):
                def __init__(self, data_list):
                    self.data_list = data_list
                def __getitem__(self, i):
                    return process_test_im(self.data_list[i])
                def __len__(self):
                    return len(self.data_list)

            test_dataset = TestDataset(test_data)
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False,
                collate_fn=lambda x: (torch.cat([a[0] for a in x], 0), torch.cat([a[1] for a in x], 0))
            )

            print("Running test loop...")
            test_loop(test_loader, net)
    else:
        parser.print_help()
