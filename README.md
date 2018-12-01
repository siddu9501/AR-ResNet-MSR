# AR-ResNet-MSR
Action Recognition using Resnet on MSR 3d Activity dataset

### Dataset
MSR Activity dataset contains real world co-ordinates and local camera co-ordinates. We make use of both of them.

- Load data for training using load_dataset()
- Requires 3 numpy files containing real world co-ordinates, local co-ordinates and the labels
- Each of the files should be a 4D-array -> (N x T x Joints x 3)

### Model
Choose model from ResNet18, ResNet34;
- ResNet50 and above do not work with embedded devices due to want for memory.

### Training
Running the train.py will run the model and return the top-1 accuracy of the model on the test and the training set.
