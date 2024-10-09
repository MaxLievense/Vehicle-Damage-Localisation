# Vehicle-Damage-Localisation
This repository presents a pipeline to address Lensor's Damage Detection challenge. The primary objective is to build an object detection pipeline that identifies various types of vehicle damages using ML. The project emphasises a structured approach to damage detection in a multi-class setting, prioritising the creation of a clean and extensible solution over achieving state-of-the-art (SOTA) performance.

## Getting started
ollow the instructions below to run the repository (assuming a Linux environment).

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
python3 src/main.py --config-name=FasterRCNN.yaml
```
Alternatively, you can customize the configuration. An example command might look like this:
```bash
python3 src/main.py \
    +data=data +data/dataset=VehicleDamage \
    +model=detection/RCNN/FasterRCNN_MNv3_L \
    +trainer=detection/DetectionTrainer +trainer/optimizer=SGD \
    +trainer/scheduler=StepLR +trainer/callbacks/eval=eval \
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
### Testing
There was not much testing required for this project, which I can discuss further in the interview. However, tests can be added to the corresponding module, as shown in `src/utils/test/test_metrics.py`, which tests the metric accumulation implementation.

### Deployment
I skipped the last step in the challenge since deployment is highly dependent on the specific use case and the platforms available. I can provide more details on this during the interview.

## Dataset
The provided dataset consists of vehicle images with damage annotations in COCO format, divided into 8 classes. In this implementation, a `min_area` filter is used to exclude annotations below a certain size. The class distribution is shown below:

| `min_area` 	| Subset 	| Total 	| 1   	| 2    	| 3   	| 4   	| 5   	| 6   	| 7   	| 8   	|
|------------	|--------	|-------	|-----	|------	|-----	|-----	|-----	|-----	|-----	|-----	|
|            	| Train  	| 1579  	| 240 	| 2525 	| 69  	| 233 	| 288 	| 130 	| 18  	| 9   	|
| None      	| Val    	| 144   	| 11  	| 104  	| 1   	| 25  	| 12  	| 4   	| 1   	| 1   	|
|            	| Test   	| 75    	| 19  	| 239  	| 3   	| 17  	| 25  	| 10  	| 3   	| 2   	|
|------------	|--------	|-------	|-----	|------	|-----	|-----	|-----	|-----	|-----	|-----	|
|            	| Train  	| 1117  	| 131 	| 805  	| 63  	| 218 	| 264 	| 130 	| 18  	| 9   	|
| 500        	| Val    	| 86    	| 3   	| 27   	| 1   	| 25  	| 12  	| 4   	| 1   	| 1   	|
|            	| Test   	| 47    	| 12  	| 65   	| 3   	| 14  	| 21  	| 10  	| 3   	| 2   	|
|------------	|--------	|-------	|-----	|------	|-----	|-----	|-----	|-----	|-----	|-----	|
|            	| Train  	| 759   	| 89  	| 333  	| 57  	| 191 	| 201 	| 130 	| 18  	| 9   	|
| 2000       	| Val    	| 57    	| 1   	| 11   	| 1   	| 24  	| 10  	| 4   	| 1   	| 1   	|
|            	| Test   	| 36    	| 1   	| 27   	| 3   	| 12  	| 15  	| 10  	| 3   	| 2   	|

From an evaluation perspective, this "long-tailed" distribution, where some classes have more examples than others, can cause biases toward the more common classes. I would aim to create a more evenly distributed test set if possible and might use k-fold cross-validation to ensure generalizability across the entire dataset.

To address the bias, especially for smaller objects, I would consider the following approaches:
* Adjusting NMS (Non-Maximum Suppression): This step removes duplicate bounding boxes and might be the reason why smaller detections are left out, especially when there are overlapping annotations.
* Rebalancing anchor ratios and scales: Ensuring that smaller boxes are included as candidate regions will increase computational costs and might require tuning the NMS threshold.
* Changing the loss function from `CrossEntropy` to Focal Loss: `Focal Loss` adds a modulation term that adjusts the loss for higher class probabilities, helping to reduce bias toward larger objects. This may also require additional tuning.
