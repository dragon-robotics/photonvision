# AprilTag ROI Source Model

`best.pt` is the source YOLO model for ML-assisted AprilTag ROI detection.

Use this file as the conversion source for Orin Nano runtime artifacts:

```text
best.pt -> ONNX -> TensorRT engine
```

The ONNX artifact should be exported for the PhotonVision runtime model manager and placed under:

```text
photon-server/src/main/resources/models/
```

Do not commit TensorRT `.engine` files unless they are proven compatible with the pinned Orin Nano
image. TensorRT engines are usually tied to the JetPack, TensorRT, CUDA, and GPU runtime.

Suggested export command:

```bash
yolo export model=models/source/apriltag/best.pt format=onnx imgsz=640 simplify=True opset=12
```
