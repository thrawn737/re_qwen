mkdir -p ./data/coco
wget http://images.cocodataset.org/zips/val2014.zip -P ./data/coco && unzip ./data/coco/val2014.zip -d ./data/coco &
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip -P ./data/coco && unzip ./data/coco/annotations_trainval2014.zip -d ./data/coco &