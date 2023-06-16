import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import time

from datasets.heron_data import HeronData
from datasets.mastr import MaSTr1325Dataset
from datasets.mods import MODSDataset
from datasets.transforms import PytorchHubNormalization
from wasr.inference import Predictor
import wasr.models as models
from wasr.utils import load_weights
from torchmetrics import JaccardIndex


# Colors corresponding to each segmentation class
SEGMENTATION_COLORS = np.array([
    [247, 195, 37],
    [41, 167, 224],
    [90, 75, 164]
], np.uint8)

BATCH_SIZE = 1
WORKERS = 1
DATASET_PATH = os.path.expanduser('heron_data')
MODEL = 'ewasr_resnet18'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="WaSR Network MaSTr1325 Inference")
    parser.add_argument("--num_classes", type=int, default=3,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--model", type=str, choices=models.model_list, default=MODEL,
                        help="Model architecture.")
    parser.add_argument("--dataset-path", type=str, default=DATASET_PATH,
                        help="Path to the MODS dataset root.")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to the model weights or a model checkpoint.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Minibatch size (number of samples) used on each device.")
    parser.add_argument("--fp16", action='store_true',
                        help="Use half precision for inference.")
    parser.add_argument("--mixer", type=str, default="CCCCSS", help="Token mixers in feature mixer.")
    parser.add_argument("--project", action='store_true', help="Project encoder features to less channels.")
    parser.add_argument("--enricher", type=str, default="SS", help="Token mixers in long-skip feature enricher.")
    parser.add_argument("--color-constant", action="store_true",
                        help="Use additional color constant input channel."),
    parser.add_argument('--export-onnx', action='store_true',
                        help="Export the model to ONNX format.")
    parser.add_argument('--trt', action='store_true',
                        help="Run inference with TensorRT")
 
    return parser.parse_args()

output_dir = get_arguments().output_dir

def export_predictions(preds, batch, output_dir=output_dir):
    features, metadata = batch
    pred_mask = SEGMENTATION_COLORS[preds]
    mask_img = Image.fromarray(pred_mask)
    gt_mask = metadata['label'].squeeze()
    gt_img = Image.fromarray(SEGMENTATION_COLORS[gt_mask])

    seq_dir = str(output_dir) + '/test'
    fig = plt.figure(figsize=(8, 16))
    orig_img = Image.open(metadata['image_path'][0])
    fig.add_subplot(3, 1, 1)
    plt.imshow(orig_img)
    if gt_mask != None:
        plt.imshow(gt_img, alpha=0.5)
    plt.axis('off')
    fig.add_subplot(3, 1, 2)
    plt.axis('off')

    fig.add_subplot(3, 1, 3)
    plt.imshow(orig_img)
    plt.imshow(mask_img, alpha=0.5)
    plt.axis('off')

    existing_files = os.listdir(seq_dir)
    existing_files.sort(key=lambda x: int(x.split('.')[0]))
    next_file_number = int(existing_files[-1].split('.')[0]) + 1 if existing_files else 1

    out_file = (seq_dir + '/' + str(next_file_number) + '.png')
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close(fig)
    
def export_gif(output_dir):
    import glob
    from pathlib import Path
    from PIL import Image

    seq_dirs = [x for x in Path(output_dir).iterdir() if x.is_dir()]
    for seq_dir in seq_dirs:
        images = []
        for filename in sorted(glob.glob(str(seq_dir / '*.png'))):
            print(f"Reading image: {filename}")
            image = Image.open(filename)
            images.append(image)
        
        if len(images) > 0:
            images[0].save(str(seq_dir / 'result.gif'), save_all=True, append_images=images[1:], duration=500, loop=0)
        else:
            print(f"No images found in directory: {seq_dir}")
            
def calculate_metrics(preds, batch, num_classes=2):
    features, metadata = batch
    gt_mask = metadata['label'].squeeze()

    if num_classes == 2:
        preds[preds==2] = 0 # sky: 2 -> 0, obstacle: 0, water: 1
        gt_mask[gt_mask==2] = 0
    jaccard = JaccardIndex(task='multiclass', num_classes=num_classes)
    iou = jaccard(preds, gt_mask)
    print(iou.item())
    return iou

def predict(args):
    # Create augmentation transform if not disabled
    dataset = HeronData(
        args.dataset_path,
        normalize_t=PytorchHubNormalization(),
        four_channel_in=args.color_constant
    )
    dl = DataLoader(dataset, batch_size=args.batch_size, num_workers=1)

    device = 'cuda' 
    # Load model
    model = models.get_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes,
        four_channel_in=args.color_constant
    )
    state_dict = load_weights(args.weights)
    model.load_state_dict(state_dict)
    model.eval().to(device)

    output_dir = Path(args.output_dir)

    wl_metric = WaterLineMetric()
    mIou = 0
    for i, frame in enumerate(dl):
        with torch.no_grad():
            features, metadata = frame
            _, C, H, W = features['image'].shape
            ts = time.time()
            features['image'] = features['image'].to(device)
            res = model(features)
            probs = res['out'].softmax(1).cpu()
            probs = torch.nn.functional.interpolate(probs, (H,W), mode='bilinear')
            preds = probs.argmax(1)[0]
            elapsed = time.time() - ts
            
            # print(metadata['seq'][0], metadata['name'][0], end=" | ")
            print(f'Inference time: {elapsed:.2f}s')

            export_predictions(preds, frame, output_dir)
            iou = calculate_metrics(preds, frame, args.num_classes)
            mIou += iou.item()

            wl_metric.update(preds, probs[0], metadata['label'][0])
    mIou /= len(dl)
    print('mIoU: ', mIou)
    print('Water line error: ', wl_metric.get_pixel_error())
    print('Water line error (lilypad): ', wl_metric.get_pixel_error_lily())
    wl_metric.plot_prcurve()

class WaterLineMetric():
    def __init__(self):
        self.pixel_error = 0
        self.pixel_error_lily = 0
        self.N = 0
        #self.pred_lines_allp = []
        #self.gt_lines = []
        self.tp = {p:0 for p in np.arange(0, 1, 0.01)}
        self.fp = {p:0 for p in np.arange(0, 1, 0.01)}
        self.fn = {p:0 for p in np.arange(0, 1, 0.01)}

    def get_water_line(self, preds, window_size=1, lilypad=False):
        height, width = preds.shape
        if lilypad:
            preds[preds>=2] = 0 # lilypad: 2 -> 0, obstacle: 0, water: 1 

        line = np.zeros(width)
        cumsum = np.cumsum(preds[::-1], axis=0)
        preds_window = cumsum[window_size:, :] - cumsum[:-window_size, :]
        for w in range(width):
            for h in range(0, height-window_size):
                #if preds[height - 1 - h, w] == 0:
                if preds_window[h, w] == 0:
                    line[w] = height - 1 -h
                    break
        return line
    
    def scan_water_line(self, preds, max_window_size = 100, lilypad=False):
        height, width = preds.shape 
        
        line = np.zeros((width, max_window_size))
        for w in range(width):
            i = 0
            count = 0
            for h in range(0, height):
                if (preds[height - 1 - h, w] == 0) or (lilypad and preds[height - 1 - h, w] == 2):
                    count += 1
                    if count > i:
                        line[w, i] = height - 1 - h
                        i += 1
                        if i >= max_window_size:
                            break
                else:
                    count = 0
        
        return line

    def count_tpfpfn(self, probs, targets):

        t = targets
        for p in np.arange(0, 1, 0.01):
            water = (probs[1, :, :] > p)
            self.tp[p] += (water[t == 1]).sum()
            self.fp[p] += (water[t == 0]).sum()
            self.fn[p] += ((~water)[t == 1]).sum()


    def update(self, preds, probs, targets):
        self.N += 1
        pred_line = self.get_water_line(preds.cpu().numpy(), window_size=1)
        target_line = self.get_water_line(targets.cpu().numpy(), window_size=1)

        pixel_error = np.mean(np.abs(pred_line - target_line)) 
        self.pixel_error += pixel_error

        pred_line = self.get_water_line(preds.cpu().numpy(), window_size=1, lilypad=True)
        target_line = self.get_water_line(targets.cpu().numpy(), window_size=1, lilypad=True)

        # pred_line_allp = self.scan_water_line(preds.cpu().numpy(), max_window_size=20)
        # self.gt_lines.append(target_line)
        # self.pred_lines_allp.append(pred_line_allp)
    
        pixel_error = np.mean(np.abs(pred_line - target_line)) 
        self.pixel_error_lily += pixel_error

        self.count_tpfpfn(probs.cpu().numpy(), targets.cpu().numpy())

    def get_pr(self, p, log=False):
        fn = self.fn[p]
        fp = self.fp[p]
        tp = self.tp[p]
        if (tp + fp) == 0:
            precision = 1
        else:
            precision = tp / (tp + fp) # precision
        if (tp + fn) == 0:
            recall = 1
        else:
            recall = tp / (tp + fn) # recall
        if log:
            print('threshold', p, 'precision: ', precision, 'recall: ', recall)
        
        return precision, recall

    def plot_prcurve(self): 
        prcurve = np.zeros((100, 2))
        for i in range(100):
            prcurve[i] = self.get_pr(i * 0.01)
        plt.figure()
        plt.plot(prcurve[:, 1], prcurve[:, 0], label='All pixel')
        plt.ylabel('precision')
        plt.xlabel('recall')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.legend()
        plt.savefig('prcurve.png')
        plt.show()

    def get_pixel_error(self):
        return self.pixel_error / self.N
    
    def get_pixel_error_lily(self):
        return self.pixel_error_lily / self.N

class ONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ONNXWrapper, self).__init__()

        self.model = model

    def forward(self, x):
        input = {'image': x}
        return self.model(input)

def convert_onnx(args):
    import torch.onnx
    import onnx

    dataset = HeronData(
        args.dataset_path,
        normalize_t=PytorchHubNormalization(),
        four_channel_in=args.color_constant
    )
    dl = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)

    model = models.get_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes,
        four_channel_in=args.color_constant
    )
    state_dict = load_weights(args.weights)
    model.load_state_dict(state_dict)
    torch.save(model, args.weights+"_model.pt")
    model.eval()
    model = ONNXWrapper(model)

    if args.color_constant:
        dummy_input = torch.randn(1, 4, 360, 640)
    else:
        dummy_input = torch.randn(1, 3, 360, 640)
    torch_out = model(dummy_input)
    onnx_output_path = args.weights + ".onnx"
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         onnx_output_path,       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX') 

    onnx_model = onnx.load(onnx_output_path)
    onnx.checker.check_model(onnx_model)
    import onnxruntime

    ort_session = onnxruntime.InferenceSession(args.weights + ".onnx", providers=['CPUExecutionProvider'])
    print(onnxruntime.get_device())

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out['out']), ort_outs[0], rtol=1e-02, atol=1e-03)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

def predict_onnx(args):
    import onnx, onnxruntime

    onnx_output_path = args.weights + ".onnx"

    onnx_model = onnx.load(onnx_output_path)
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(args.weights + ".onnx", providers=['TensorrtExecutionProvider'])
    print(onnxruntime.get_device())

    dummy_input = np.random.randn(1, 3, 360, 640).astype(np.float32)

    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs)

def predict_trt(args):
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    
    logger = trt.Logger(trt.Logger.WARNING)
    
    trt_weights = args.weights + ".trt"

    with open(trt_weights, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()

        # need to set input and output precisions to FP16 to fully enable it
        if args.color_constant:
            input_batch = np.random.randn(1, 4, 360, 640).astype(np.float32)
        else:
            input_batch = np.random.randn(1, 3, 360, 640).astype(np.float32)
        output1 = np.empty([1, 1024, 45, 80], dtype=np.float32)
        output2 = np.empty([1, 3, 90, 160], dtype = np.float32) 

        # allocate device memory
        d_input = cuda.mem_alloc(1 * input_batch.nbytes)
        d_output1 = cuda.mem_alloc(1 * output1.nbytes)
        d_output2 = cuda.mem_alloc(1 * output2.nbytes)

        bindings = [int(d_input), int(d_output1), int(d_output2)]

        stream = cuda.Stream()
        # transfer input data to device
        cuda.memcpy_htod_async(d_input, input_batch, stream)
        # execute model
        context.execute_async_v2(bindings, stream.handle, None)
        # transfer predictions back
        cuda.memcpy_dtoh_async(output2, d_output2, stream)
        # syncronize threads 
        stream.synchronize()
        print(output2)


def main():
    args = get_arguments()
    print(args)

    if args.export_onnx:
        convert_onnx(args)
        return
    if args.trt:
        predict_trt(args)
    else:
        predict(args)

        export_gif(args.output_dir)


if __name__ == '__main__':
    main()
