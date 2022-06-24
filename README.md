# alexnet pytorch 

:octocat: re-implementation of alexnet  

### comparision with torchvision implementations    

experiments torchvision pretrained params of imagenet classification in ubuntu

### Environments

- Python 3.8
- pytorch == 1.10.2 
- torchvision == 0.11.3 

### Experiment via torchvision pretrained params at validation set

test_torchvision.py 

MFLOPS - 10^6

GFLOPS - 10^9

TFLOPS - 10^12

https://pytorch.org/vision/stable/models.html

### Experiment via our training params at validation set

Imagenet

|model                 | # parameters      | Flops              | Resolution | top-1 Acc | top-5 Acc | top-1 Err | top-5 Err | epoch |
|----------------------|-------------------| ------------------ | ---------- | --------- |-----------|-----------| ----------|-------| 
|alexnet (paper)       | -                 | -                  | 224 x 224  | 56.522    | 79.066    | 38.1      | 16.4      |  -    |
|alexnet (torchvision) | 61100840          | 714691904          | 224 x 224  | 56.516    | 79.070    | 43.484    | 20.93     |  _    |
|alexnet (this-repo)   | 61100840          | 714691904          | 224 x 224  | 55.940    | 78.894    | 44.060    | 21.106    |  72   |

### training options

- batch : 128
- scheduler : step LR
- loss : cross-entropy
- dataset : imagenet ~(138M)
- epoch : 90
- gpu : nvidia geforce rtx 3090 x 4EA
- lr : 1e-2

### training

- dataset

    train : Imagenet training dataset
    test : Imagenet validation dataset

- data augmentation

    for training

    1. RandomResizedCrop()
    2. RandomHorizontalFlip()
    3. PCANoisePIL()
    4. ToTensor()
    5. normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    for testing
    
    1. Resize()
    2. CenterCrop()
    3. ToTensor()
    4. normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
### Reference

Alexnet : https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf


### Start Guide


