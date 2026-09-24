"""Run one image through the SAR, RGB or IR branch; save detections as JSON."""
import argparse
import json
from pathlib import Path

import torch
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose
from runtime import MODALITIES, load_config, load_model, setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('image')
    parser.add_argument('--modality', choices=MODALITIES, required=True)
    parser.add_argument('--out', default='detections.json')
    args = parser.parse_args()
    setup()
    cfg = load_config(args.config)
    model = load_model(cfg, args.checkpoint)
    pipeline = Compose(cfg.pipelines[args.modality])
    data = pipeline(dict(img_info=dict(filename=str(Path(args.image).resolve())),
                         img_prefix=None))
    data = scatter(collate([data], samples_per_gpu=1), [0])[0]
    with torch.no_grad():
        prediction = model(return_loss=False, rescale=True, **data)[0]
    # SAR: [x1,y1,x2,y2,score]. RGB/IR: [cx,cy,w,h,angle_radians,score].
    result = {'modality': args.modality,
              'box_format': 'xyxy_score' if args.modality == 'sar'
                            else 'cxcywh_angle_radians_score',
              'detections': {name: boxes.tolist()
                             for name, boxes in zip(model.CLASSES, prediction)}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(out)


if __name__ == '__main__':
    main()
