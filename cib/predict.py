from fastai import *
from fastai.vision import *
import json

class Predict:
    
    def __init__(self):
        self.data_path = Path("DATA")
        
        with open('classes.json') as f:
            self.classes = json.load(f)
            
            data = ImageDataBunch.single_from_classes(
                         self.data_path, self.classes, 
                         tfms=get_transforms(), size=224).normalize(imagenet_stats)
            
            self.learn = create_cnn(data, models.resnet34, metrics=error_rate, pretrained=False)
            self.learn.load('stage-5')
    
    def predict(self,stream):
        img = open_image(stream)
        pred_class,pred_idx,outputs = self.learn.predict(img)
        
        probs = outputs.numpy()
        probs = probs / probs.sum() * 100.0

        df = pd.DataFrame({ 'LABELS' : self.classes,
                            'PROBS' : probs })
        df = df.sort_values(['PROBS'], ascending=[False])
        df = df.head(5)
        
        return df