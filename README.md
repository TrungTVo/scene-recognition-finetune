# Scene recognition fine-tune
- Fine-tuned and pre-trained deep ConvNet on scene dataset.
- Training AlexNet on GPU takes approximately 7 mins and achieved result is ~73.7% accuracy.
- Data augmentation and other pretrained model can be implemented to further improve the result.

Dataset:
<img src="dataset.png">

## Training model
- Follow different Pytorch pretrained models in this link: http://pytorch.org/docs/master/torchvision/models.html
- Transfer learning tutorial using Pytorch can be found at: http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

## Usage
```
python fine_tune.py
```

## Results
Below is the result when train on GPU with AlexNet (~73.7% accuracy) and train from scratch (~49% accuracy).

<img width="323" alt="finetune" src="https://user-images.githubusercontent.com/20756728/38471358-1e901dc6-3b3e-11e8-9f7c-d37b9d4bf307.png" description="trained on AlexNet"> <img width="314" alt="train_scratch" src="https://user-images.githubusercontent.com/20756728/38471360-28165c02-3b3e-11e8-871d-13b091afbc5c.png" description="trained from scratch">
