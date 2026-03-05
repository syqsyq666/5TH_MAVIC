#!/usr/bin/env python3
"""
统一训练入口：可选择训练 ResNet101（norm_resnet50_SAR）或 EfficientNet-B0（efficient_SAR），或两者依次训练。

用法:
  python run_train.py --model resnet      # 只训练 ResNet101（EO+SAR 跨域）
  python run_train.py --model efficient   # 只训练 EfficientNet-B0
  python run_train.py --model both        # 先 ResNet101，再 EfficientNet-B0
  python run_train.py                     # 默认 both
"""
import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description='CSRN PBVS2025 训练：ResNet101 / EfficientNet-B0')
    parser.add_argument('--model', type=str, default='both',
                        choices=['resnet', 'efficient', 'both'],
                        help='resnet=norm_resnet50_SAR, efficient=efficient_SAR, both=两者依次训练')
    parser.add_argument('--gpus', type=str, default='0,1',
                        help='GPU 编号，逗号分隔，如 0,1')
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.gpus

    scripts = []
    if args.model in ('resnet', 'both'):
        scripts.append(('ResNet101 (norm_resnet50_SAR)', 'norm_resnet50_SAR.py'))
    if args.model in ('efficient', 'both'):
        scripts.append(('EfficientNet-B0 (efficient_SAR)', 'efficient_SAR.py'))

    for name, script in scripts:
        path = os.path.join(PROJECT_ROOT, script)
        if not os.path.isfile(path):
            print(f'未找到脚本: {path}')
            continue
        print('=' * 60)
        print(f'开始训练: {name}')
        print('=' * 60)
        ret = subprocess.run([sys.executable, path], cwd=PROJECT_ROOT, env=env)
        if ret.returncode != 0:
            print(f'训练脚本 {script} 退出码: {ret.returncode}')
            sys.exit(ret.returncode)

    print('全部训练流程结束。')
    print('ResNet 权重目录: checkpoints/resnet101/')
    print('EfficientNet 权重目录: checkpoints/efficientnet_b0/')
    print('测试时可将 test.py 中模型路径指向上述目录下对应 .pth 文件。')


if __name__ == '__main__':
    main()
