# Deploy
Change working directory to `TGRMPT/deploy`.

### detection

To deploy our method on board of a robot using TensorRT, first, deploy whole body and head shoulder detection models (see [detection README](../detection/README.md)), 
and copy the generated engine models into the folder `detection/weights`, and be sure to rename them to `head_shoulder.engine` and `whole_body.engine`.



/home/lzy-local/wkspace/tgrmpt/detection/yolov5/runs/wb/exp/weights/best.onnx



### reid

Second, deploy whole body and head shoulder ReID models (see [reid README](../reid/README.md)), and copy the generated engine models into the folder `embedding/weights`, 
and be sure to rename them to `head_shoulder.engine` and `whole_body.engine`.

logs/bot_r18_train_on_iros2022_fisheye_whole_body_hs_mask_layer_4x2/whole_body.onnx

### mot

To run our method on a sequence of images, e.g., `../dataset/mot/mot17/02_original_black_fisheye_head_front`, run

```shell
python3 deep_sort_app.py --sequence_dir ../dataset/mot/mot17/02_original_black_fisheye_head_front/img1
```
Run `python deep_sort_app.py --help` to see more options.

.deploy_trt.sh.swp



```
python3 deep_sort_app.py --sequence_dir  ../datasets/02_original_black_fisheye_head_front/img1


/home/lzy-local/wkspace/tgrmpt/tracking/eval/data/gt/zjlab/iros2022-fisheye-original-test/02_original_black_fisheye_head_front/img1
```

