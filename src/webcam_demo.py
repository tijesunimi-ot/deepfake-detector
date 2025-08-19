# src/webcam_demo.py
import cv2
import numpy as np
import argparse
from facenet_pytorch import MTCNN
import onnxruntime as ort
from src.infer_onnx import preprocess_img_bgr, IMAGENET_MEAN, IMAGENET_STD

def choose_largest_box(boxes):
    if boxes is None or len(boxes) == 0:
        return None
    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    idx = int(np.argmax(areas))
    return boxes[idx]

def crop_square_from_box(img, box, scale=1.2):
    h,w = img.shape[:2]
    x1,y1,x2,y2 = box
    cx = int((x1+x2)/2)
    cy = int((y1+y2)/2)
    half = int(max(x2-x1, y2-y1)/2 * scale)
    left = max(cx-half, 0)
    top  = max(cy-half, 0)
    right = min(cx+half, w)
    bottom= min(cy+half, h)
    return img[top:bottom, left:right], (left,top,right,bottom)

def run_demo(onnx_path, cam_index=0, size=160, use_cuda=False, provider_preference=None):
    # MTCNN detector
    mtcnn = MTCNN(keep_all=True, device='cuda' if use_cuda else 'cpu')
    providers = provider_preference or (["CUDAExecutionProvider","CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"])
    sess = ort.InferenceSession(str(onnx_path), providers=providers)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise SystemExit("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # detect faces (returns numpy arrays of boxes)
        boxes, probs = mtcnn.detect(frame)
        chosen = choose_largest_box(boxes) if boxes is not None else None
        out_frame = frame.copy()
        prob_fake = None

        if chosen is not None:
            crop, bbox = crop_square_from_box(frame, chosen, scale=1.2)
            if crop.size != 0:
                inp = preprocess_img_bgr(crop, size=size)
                inp = np.expand_dims(inp, axis=0).astype(np.float32)
                input_name = sess.get_inputs()[0].name
                out_name = sess.get_outputs()[0].name
                logits = sess.run([out_name], {input_name: inp})[0].squeeze()
                exp = np.exp(logits - np.max(logits))
                probs = exp / exp.sum()
                prob_fake = float(probs[1])
                # draw bbox and text
                l,t,r,b = bbox
                cv2.rectangle(out_frame, (l,t), (r,b), (0,255,0), 2)
                cv2.putText(out_frame, f"fake:{prob_fake:.3f}", (l, max(10,t-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Deepfake Detector (press q to quit)", out_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--use_cuda", action="store_true")
    args = ap.parse_args()
    run_demo(args.onnx, cam_index=args.cam, size=args.size, use_cuda=args.use_cuda)

#command to run: python src/webcam_demo.py --onnx checkpoints/onnx/mobilenet_v2.onnx --use_cuda
# or use CPU:
# python src/webcam_demo.py --onnx checkpoints/onnx/mobilenet_v2.onnx
