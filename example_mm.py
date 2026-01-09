import torch
import argparse
import numpy as np
import os
from pi3.utils.basic import load_multimodal_data, write_ply
from pi3.utils.geometry import depth_edge
from pi3.models.pi3x import Pi3X

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run inference with the Pi3 model.")
    
    parser.add_argument("--data_path", type=str, default='examples/skating.mp4',
                        help="Path to the input image directory or a video file.")
    
    # parser.add_argument("--conditions_path", type=str, default='examples/room/condition.npz',
    parser.add_argument("--conditions_path", type=str, default=None,
                        help="Optional path to a .npz file containing 'poses', 'depths', 'intrinsics'.")

    parser.add_argument("--save_path", type=str, default='examples/result.ply',
                        help="Path to save the output .ply file.")
    parser.add_argument("--save_transforms", type=str, default=None,
                        help="Path to save transforms.json with camera poses and intrinsics. Default: None (not saved)")
    parser.add_argument("--use_moge_intrinsics", action='store_true',
                        help="Use MoGe method to recover intrinsics from local_points. Default: False (use input or default)")
    parser.add_argument("--interval", type=int, default=-1,
                        help="Interval to sample image. Default: 1 for images dir, 10 for video")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to the model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
                        
    args = parser.parse_args()
    if args.interval < 0:
        args.interval = 10 if args.data_path.endswith('.mp4') else 1
    print(f'Sampling interval: {args.interval}')

    # 1. Prepare model
    print(f"Loading model...")
    device = torch.device(args.device)
    if args.ckpt is not None:
        model = Pi3X().to(device).eval()
        if args.ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(args.ckpt)
        else:
            weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        
        model.load_state_dict(weight, strict=False)
    else:
        model = Pi3X.from_pretrained("yyfz233/Pi3X").to(device).eval()
        # or download checkpoints from `https://huggingface.co/yyfz233/Pi3X/resolve/main/model.safetensors`, and `--ckpt ckpts/model.safetensors`

    # 2. Prepare input data

    # Load optional conditions from .npz
    poses = None
    depths = None
    intrinsics = None

    if args.conditions_path is not None and os.path.exists(args.conditions_path):
        print(f"Loading conditions from {args.conditions_path}...")
        data_npz = np.load(args.conditions_path, allow_pickle=True)

        poses = data_npz['poses']             # Expected (N, 4, 4) OpenCV camera-to-world
        depths = data_npz['depths']           # Expected (N, H, W)
        intrinsics = data_npz['intrinsics']   # Expected (N, 3, 3)

    conditions = dict(
        intrinsics=intrinsics,
        poses=poses,
        depths=depths
    )

    # Load images (Required)
    imgs, conditions = load_multimodal_data(args.data_path, conditions, interval=args.interval, device=device) 

    """
    Args:
        imgs (torch.Tensor): Input RGB images valued in [0, 1].
            Shape: (B, N, 3, H, W).
        intrinsics (torch.Tensor, optional): Camera intrinsic matrices.
            Shape: (B, N, 3, 3).
            Values are in pixel coordinates (not normalized).
        rays (torch.Tensor, optional): Pre-computed ray directions (unit vectors).
            Shape: (B, N, H, W, 3).
            Can replace `intrinsics` as a geometric condition.
        poses (torch.Tensor, optional): Camera-to-World matrices.
            Shape: (B, N, 4, 4).
            Coordinate system: OpenCV convention (Right-Down-Forward).
        depths (torch.Tensor, optional): Ground truth or prior depth maps.
            Shape: (B, N, H, W).
            Invalid values (e.g., sky or missing data) should be set to 0.
        mask_add_depth (torch.Tensor, optional): Mask for depth condition.
            Shape: (B, N, N).
        mask_add_ray (torch.Tensor, optional): Mask for ray/intrinsic condition.
            Shape: (B, N, N).
        mask_add_pose (torch.Tensor, optional): Mask for pose condition.
            Shape: (B, N, N).
            Note: Requires at least two frames to be True to establish a meaningful
            coordinate system (absolute pose for a single frame provides no relative constraint).
    """

    # 3. Infer
    print("Running model inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            res = model(
                imgs=imgs, 
                **conditions
            )

    # 4. process mask
    masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
    non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
    masks = torch.logical_and(masks, non_edge)[0]

    # 5. Save points
    print(f"Saving point cloud to: {args.save_path}")
    if os.path.dirname(args.save_path):
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        
    write_ply(res['points'][0][masks].cpu(), imgs[0].permute(0, 2, 3, 1)[masks], args.save_path)
    print("Done.")
    
    # 6. Save transforms.json (optional)
    if args.save_transforms:
        print("\n" + "="*60)
        print("保存相机位姿和内参到 transforms.json...")
        print("="*60)
        
        from pi3.utils.transforms_utils import save_transforms_json, generate_image_paths
        from PIL import Image
        
        # 提取位姿 (OpenCV camera-to-world)
        camera_poses = res['camera_poses'][0].cpu().numpy()  # (N, 4, 4)
        N = camera_poses.shape[0]
        H, W = imgs.shape[-2:]
        
        # 获取内参
        intrinsics_np = None
        if conditions.get('intrinsics') is not None:
            intrinsics_np = conditions['intrinsics'][0].cpu().numpy()  # (N, 3, 3)
            print(f"使用输入的内参")
        
        # 生成图像路径
        image_paths = generate_image_paths(args.data_path, N, save_dir='images')
        
        # 保存 resize 后的图片（先保存图片，再生成正确的路径）
        output_dir = os.path.dirname(args.save_transforms) or '.'
        images_dir = os.path.join(output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        print(f"保存 resize 后的图片到: {images_dir}")
        imgs_np = imgs[0].cpu().numpy()  # (N, 3, H, W)
        
        # 保存图片并更新路径
        updated_image_paths = []
        for i in range(N):
            # 提取文件名（从 image_paths 中获取，如 "./images/frame_0000.png"）
            img_filename = os.path.basename(image_paths[i])
            
            # 对于非标准扩展名（如 .heic），统一保存为 .png
            name_without_ext, ext = os.path.splitext(img_filename)
            if ext.lower() in ['.heic', '.jpeg']:
                img_filename = name_without_ext + '.png'
            elif ext.lower() == '.jpg':
                img_filename = name_without_ext + '.jpg'  # 保持 jpg
            else:
                img_filename = name_without_ext + '.png'  # 默认 png
            
            save_path = os.path.join(images_dir, img_filename)
            
            # 转换为 PIL Image (0-1范围转为0-255)
            img_array = imgs_np[i].transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)
            img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_array)
            
            # 保存
            img_pil.save(save_path)
            
            # 更新路径
            updated_image_paths.append(f"./images/{img_filename}")
        
        print(f"已保存 {N} 张图片到 {images_dir}")
        
        # 保存 transforms.json（使用更新后的路径）
        save_transforms_json(
            output_path=args.save_transforms,
            camera_poses=camera_poses,
            intrinsics=intrinsics_np,
            image_paths=updated_image_paths,
            image_size=(H, W),
            res=res if args.use_moge_intrinsics else None,
            imgs=imgs if args.use_moge_intrinsics else None,
            use_moge_recovery=args.use_moge_intrinsics
        )
        print("="*60 + "\n")