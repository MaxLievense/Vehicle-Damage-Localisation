# Vehicle-Damage-Localisation
This repository presents a pipeline to address Lensor's Damage Detection challenge. The primary objective is to build an object detection pipeline that identifies various types of vehicle damages using ML. The project emphasises a structured approach to damage detection in a multi-class setting, prioritising the creation of a clean and extensible solution over achieving state-of-the-art (SOTA) performance.

## Getting started
Follow the instructions below to run the repository (assuming a Linux environment).

### Python Installation
Run the following commands or their equivalent:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data prepration
The `data/` directory should be structured as shown below, and can be set up by extracting the contents of the provided `.zip` file into the `data/` directory:
```
data
  └──vehicle_damage_detection
     ├──annotations
     └──images
        ├──test
        ├──train
        └──val
```  

### Running experiment
The following command will run a complete training and testing cycle:
```bash
python3 src/main.py --config-name=FasterRCNNv2.yaml
```
Alternatively, you can customize the configuration. An example command might look like this:
```bash
python3 src/main.py \
    +data=data +data/dataset=VehicleDamage \
    +model=detection/RCNN/FasterRCNN_MNv3_L \
    +trainer=detection/DetectionTrainer +trainer/optimizer=SGD \
    +trainer/scheduler=StepLR +trainer/callbacks/eval=eval
```
You might need to adjust `device=cuda:0` to `device=cpu` or your equivalent, depending on your enviroment.

Results of example runs can be found [here](https://api.wandb.ai/links/maxlievense/jp1xvkma).


## Approach
**_NOTE:_** _"Your solution is not expected to have every possible feature and also doesn’t need to be perfect. More important is for it to work as expected and highlight your strengths"_

In line with these instructions, I have focused on the abstraction of the ML pipeline rather than achieving high inference accuracy. While I am fully capable of obtaining desired results using an academic approach, the main constraints for this work were limited computational resources and the allocated time frame.

### Model architecture
Using a "NVIDIA GeForce RTX 3050 Laptop GPU" with 4GB of memory would not allow the use of `ResNet50` models. Consequently, `MobileNetV3` was chosen. For simplicity and to use pretrained models, I utilized the options provided by `torchvision`. Furthermore, due to computational resources limitations, this work focused on object detection rather than segmentation tasks.

### Pipeline overview
The pipeline I am presenting is designed to support multi-domain tasks (e.g., classification, detection, segmentation). It is far from done, but it should illustrate the key concepts. 

There are three main modules (trainer, model, and data), each intended to be interchangeable. Currently, this functionality is not automated, but the idea is that you should be able to specify the task (e.g., classification, detection, or segmentation), and the modules would adjust accordingly. This means that switching from detection to segmentation should only require changing a single configuration variable, and the pipeline would automatically select the appropriate classes for each module.
#### Hydra
To implement the automatic module selection, "Hydra" can be used. Hydra appends yaml configurations in runtime and initializes the modules according to the provided configurations.

For example (`src/model/detection/RCNN/FasterRCNN_MNv3_L.yaml`):

```yaml
_target_: src.model.detection.RCNN.FasterRCNN.FasterRCNN

clip_grad: 0.25
network:
  _target_: torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn
  weights: DEFAULT
```
Running `python3 -u src/main.py +model=detection/RCNN/FasterRCNN_MNv3_L` will append this configuration under `model` in the final executed config. Then, using `instantiate(cfg.model.network)` will import the `_target_` value and initialize it with the remaining parameters as `kwargs`.

I have used Hydra across multiple projects and find it very straightforward. It also encourages structured code when used correctly.

The benefits are:

* Config-based experiments: Every experiment records the exact configurations used, making it easy to replicate results (assuming package versioning is handled, which pairs well with MLOps and Docker).
* Predefined configurations with command-line tweaks: Modify certain variables through the command line as needed.
* Easily swap modules or configurations: Module or configuration switching can be done through the command line.
* Optimized memory usage: Only required modules are imported, preventing memory overload from importing the entire project structure when only a small part is used.
* Supports multiple ML frameworks (Torch, TensorFlow, etc.): Different frameworks can be used under the same structure, provided appropriate interfaces are defined.

### Experiment tracking
I used "Weights and Biases" for experiment tracking as I am familiar with the platform, and it’s free to use. A report showing the training, validation, and testing plots, along with visualizations of the outputs, can be found [here](https://api.wandb.ai/links/maxlievense/jp1xvkma). By default, `wandb` is `disabled`, but it can be enabled by including `wandb.mode=online` in your command line (assuming you have set up your account).

### CI/CD
A quick GitActions CI/CD implementation has been included to that validates Pull Requests with PyLinting and testing.

### Deployment
I skipped the last step in the challenge since deployment is highly dependent on the specific use case and the platforms available. I can provide more details on this during the interview.

## Dataset
The provided dataset consists of vehicle images with damage annotations in COCO format, divided into 8 classes. An analysis has been done on how best to approach the Anchors for object detection, which can be found in this [ipynb](visualisation/data/dataloader.ipynb). Although the an `AnchorGenerator` has been implemented, it would require an custom backbone as the feature size should match the anchor space:
```bash
AssertionError: Anchors should be Tuple[Tuple[int]] because each feature map could potentially have different sizes and aspect ratios. There needs to be a match between the number of feature maps passed and the number of sizes / aspect ratios specified.
```
This approach was discontinued, due to time limitations.