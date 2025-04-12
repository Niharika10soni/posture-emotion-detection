import argparse
from pipeline import run_inference_webcam, run_batch_inference

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['infer', 'batch'], required=True, help="Choose mode: 'infer' for webcam, 'batch' for dataset test")
    parser.add_argument('--dataset', type=str, default='mini_dataset', help="Path to dataset (only used for batch mode)")
    args = parser.parse_args()

    if args.mode == 'infer':
        run_inference_webcam()
    elif args.mode == 'batch':
        run_batch_inference(dataset_path=args.dataset)
