# Pix2Pix GAN for Human Face Generation from Canny Edges

This repository contains a Pix2Pix Generative Adversarial Network (GAN) model that performs image-to-image translation, specifically translating Canny edges of a human face into the corresponding realistic human face. The project is implemented using PyTorch and is trained on the "Edges2Human" dataset.

## Acknowledgments

I would like to express my gratitude to the authors of the [Pix2Pix paper](https://arxiv.org/abs/1611.07004) titled *"Image-to-Image Translation with Conditional Adversarial Networks"*. This foundational work provided the core ideas and architecture that made this project possible.

Additionally, I would like to thank the author of the [Medium blog post](https://medium.com/@Skpd/pix2pix-gan-for-generating-map-given-satellite-images-using-pytorch-6e50c318673a) titled *"Pix2Pix GAN for Generating Map Given Satellite Images using PyTorch"*. This blog was instrumental in helping me understand the practical implementation of Pix2Pix GANs in PyTorch.

## Dataset

Refer to the dataset i have created and used in this project is the "Edges2Human Dataset" from Kaggle, which contains pairs of Canny edge maps and the corresponding human face images. You can find the dataset here: [Edges2Human Dataset](https://www.kaggle.com/datasets/seifyasserahmed/edges2human-dataset).

### Example from Dataset
Below is an example of the dataset used, where:

- **Image A** is the target image (the real human face)
- **Image B** is the input image (the Canny edge map of the face)

![Dataset Example](https://github.com/user-attachments/assets/026dd319-b0fd-4a02-a10c-27fd8ba154b1)


## Model Architecture

The Pix2Pix GAN consists of two main components:
1. **Generator**: Translates Canny edges to human faces.
2. **Discriminator**: Evaluates the realism of generated images compared to real human faces.

The model uses a U-Net-based generator and a PatchGAN-based discriminator.

## Data Preprocessing

Each image in the dataset is resized to `256x256`. Data augmentation techniques such as random horizontal flips are used during training to improve model generalization.

## Hyperparameters

The following hyperparameters were used for training the model:

- `NUM_EPOCHS: 50`
- `Learning Rate (LR): 1e-4`
- `Betas: (0.5, 0.999) (for Adam optimizer)`
- `LAMBDA: 100 (Weight for L1 loss)`
- `Image Height: 256`
- `Image Width: 256`
- `Batch Size: 16`

## Training

The model is trained for 3 epochs using the Adam optimizer. The objective function includes both the adversarial loss from the discriminator and an L1 loss between the generated and real images, weighted by the `LAMBDA` hyperparameter.

## Results After the First Epoch
After the first epoch of training, the model begins generating reasonable results, though the quality improves as training continues.

Here is a visual comparison of the input (Canny edge map), the generated image, and the target image after the first epoch:<br>

![image](https://github.com/user-attachments/assets/f43e3918-87f3-4249-b220-da0141a8736b)


> Left: Canny edge map (input) <br>
> Middle: Generated face (output) <br>
> Right: Real face (target) <br>

You can view more generated images after further epochs in the [`results/`](results/) folder of this repository.

```bash
python train.py --epochs 50 --lr 1e-4 --betas 0.5 0.999 --lambda 100 --img_height 256 --img_width 256 --batch_size 16
