# Keep this import as it works in your setup
from data_utils.IntrADataLoader import IntrADataLoader 
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=True, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    
    # Set num_category to 2 for binary classification (IntrA has two classes)
    parser.add_argument('--num_category', default=2, type=int, choices=[2], help='training on IntrA dataset (binary classification)')
    
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    return parser.parse_args()


def test(model, loader, num_class=2, vote_num=1, show_progress=False):
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))

    # Conditionally use tqdm progress bar
    loader_enum = tqdm(enumerate(loader), total=len(loader)) if show_progress else enumerate(loader)

    for j, (points, target) in loader_enum:
        points = points.to('cuda' if not args.use_cpu else 'cpu')
        target = target.to('cuda' if not args.use_cpu else 'cpu')

        points = points.transpose(2, 1)
        vote_pool = torch.zeros(target.size()[0], num_class).to('cuda' if not args.use_cpu else 'cpu')

        for _ in range(vote_num):
            pred, _ = classifier(points)
            vote_pool += pred
        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1]

        # Optionally remove print statement
        #print(f"Batch {j+1}: Predicted: {pred_choice.cpu().numpy()}, Actual: {target.cpu().numpy()}")
        prediction= pred_choice.cpu().numpy()[0]
        print(prediction)

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    if not args.use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    

    # Change dataset path to point to your IntrA dataset
    data_path = '/content/'  # Path to IntrA dataset
    
    # Replace ModelNetDataLoader with your dataset loader
    test_dataset = IntrADataLoader(root=data_path, num_point=args.num_point, num_category=args.num_category, use_uniform_sample=args.use_uniform_sample, use_normals=args.use_normals,state='test', split='test', process_data=False)
    num_workers = 0 if args.use_cpu else 10  # Reduce workers for CPU
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    '''MODEL LOADING'''
    num_class = args.num_category  # num_class is now 2 for binary classification
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if args.use_cpu:
        classifier = classifier.cpu()  # Load model to CPU
        checkpoint = torch.load('/content/IntrA-3D-Objects-Classification/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth', map_location=torch.device('cpu'))  # Load checkpoint on CPU
    else:
        classifier = classifier.cuda()
        checkpoint = torch.load('/content/IntrA-3D-Objects-Classification/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth')

    classifier.load_state_dict(checkpoint['model_state_dict'])
     
    with torch.no_grad():
        test(classifier.eval(), testDataLoader, vote_num=args.num_votes, num_class=num_class,show_progress=False)
       
if __name__ == '__main__':
    args = parse_args()
    main(args)