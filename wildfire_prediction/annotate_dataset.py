import torch
import numpy as np
from wildfire_prediction.models.ensemble import Ensemble
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from tqdm import tqdm


_transforms = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32),
        # Computed over the whole train x test x valid datasets
        transforms.Normalize(
            [75.25387867, 88.2539294, 64.06851075],
            [49.79832831, 42.03286918, 42.90422537],
        ),
        transforms.Resize((224, 224)),
    ]
)


def annotate_dataset(
        dataset_path: str, 
        output_path: str, 
        annotator: Ensemble, 
        threshold: float =0.5,
        uncertainty_threshold: float=0.0001):
    
    data = np.loadtxt(dataset_path, dtype=str, delimiter=";") 
    labeled_data = []
 
    for item in tqdm(data):
        path = item[0]
        img = Image.open(path)

        # Avoid corrupted images
        try:
            img = _transforms(img)
        except:
            continue
        
        img = img.unsqueeze(0) # Add batch dimension
        prediction, variance = annotator.predict_with_uncertainty(img)

        print(prediction, variance)

        if torch.mean(variance) < uncertainty_threshold: 
            # Annotate only if predictions are similar
            if torch.mean(prediction) > threshold:
                label = '1'
            else:
                label = '0'

            labeled_data.append([path, label])

    np.savetxt(output_path, labeled_data, delimiter=";", fmt='%s')  
