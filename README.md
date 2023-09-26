# Dreambooth Fine Tuning
## Introduction

In recent years, there have been significant advancements in AI image generation models. One such groundbreaking model is Stable Diffusion, an open-source text-to-image model that was introduced in 2022 through deep learning techniques. This model allows us to generate images based on textual descriptions. In other words, we can convert the text we provide as input into a visual representation.

In computer vision, we have too many different fields and image synthesis is one of them and it is the most hyped one nowadays. Gives outstanding results but it demands too much computational resources for training. In order to lower the computational demand, the team that introduced latent diffusion proposed an explicit separation of the compressive from the generative learning phase. This is done by an auto-encoding model which earns a space that is perceptually equivalent to the image space, but offers significantly reduced computational complexity. In this document, there is no need for detailed information about the model. For more detailed explanation of the model architecture, please see the first paper given in the references. 

As mentioned previously, these models demand too much computational resource. Also, these models lack of having different concepts with a given reference set. In the computer vision field, we have a term called fine-tuning to improve to model in desired way. So, to tackle the issue mentioned in the last sentence, recently a work is published called Dreambooth. With Dreambooth, one can easily personalize a text-to-image model with only using 3-5 images. In this work, fine- tuning is divided into two steps: 

1) fine tuning the low-resolution text-to-image model with the input images paired with a text prompt containing a unique identifier and the name of the class the subject
1) fine-tuning the super resolution components with pairs of low-resolution and high-resolution images taken from the input images set

For more detailed description of this fine-tuning technique, please see the second paper given in the references.

Before continuing with the details, there are some links for you to try training and inference. Also, a demo link and a link for the trained models by myself.
* Tutorial Notebook: <a href="https://colab.research.google.com/github/nuwandda/dreambooth-fine-tuning/blob/main/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
* [Trained Models][1]
* [Online Demo][2]

## Organization of the Document

The organization of the document is as follows:

- Chapter **Setup** gives details about how to set up the environment to fine-tune a model using Dreambooth.
- Chapter **Dataset** explains the requirements for high-quality training datasets and the concepts.
- Chapter **Training e**xplains the steps involved in training the Stable Diffusion model using Dreambooth.
- Chapter **Inference** explains the process of generating personalized images using the fine-tuned model.
- Chapter **Discussion** summarizes the key concepts and explains the advantages and challenges of using Dreambooth.

## Setup

This case study is based on a different version of Dreambooth. The forked version is developed by **ShivamShrirao**. Please visit this [link][3] to see the forked repository. As mentioned above, the demand for computational resource is too much and the reason of using a forked repo is reducing the VRAM usage. Normally, it requires about 16 GB to 24 GB in order to fine-tune the model. The maintainer optimized the code to reduce VRAM usage to under 16GB. It is also possible to fine tune a model only using 10 GB with the help of this repository.

Below, you can see the system specifications.

- Google Colab
- Python 3
- 12.7 RAM
- Tesla T4 GPU

In this section, the detailed description of some of the libraries used in this case study will be presented. Not all libraries need to be explained because some of them are basic libraries used in Stable Diffusion.

The steps for setting up for the training and inference is explained in the notebook. Therefore, you can follow each step and test the system.

### Bitsandbytes

The bitsandbytes is a lightweight wrapper around CUDA custom functions, in particular 8- bit optimizers, matrix multiplication (LLM.int8()), and quantization functions. This is an optional package. Using this package can help us to reduce the VRAM usage further. The only limitation is the version of CUDA. It only supports CUDA versions 10.2–11.7 and your machine must fulfill the following requirements:

- LLM.int8(): NVIDIA Turing (RTX 20xx; T4) or Ampere GPU (RTX 30xx; A4-A100); (a GPU from 2018 or older).
- 8-bit optimizers and quantization: NVIDIA Maxwell GPU or newer (>=GTX 9XX).

You can see the features of the library below.

- LLM.int8() inference 
- 8-bit Optimizers: Adam, AdamW, RMSProp, LARS, LAMB, Lion (saves 75% memory) 
- Stable Embedding Layer: Improved stability through better initialization, and normalization 
- 8-bit quantization: Quantile, Linear, and Dynamic quantization 
- 8-bit Matrix multiplication with mixed precision decomposition 
- Fast quantile estimation: Up to 100x faster than other algorithms

### xFormers

Xformers is a customizable building blocks. These building blocks can be used without boilerplate code. PyTorch is not using this library in default version. It has some cutting-edge techniques. The reason behind using this library is we want to reduce VRAM usage as much as possible. xFormers contains its own CUDA kernels, but dispatches to other libraries when relevant. Hence, we can reduce the VRAM usage. You can see some optimized blocks below.

- Memory-efficient exact attention - up to 10x faster 
- sparse attention 
- block-sparse attention 
- fused softmax 
- fused linear layer 
- fused layer norm 
- fused dropout(activation(x+bias)) 
- fused SwiGLU 

### Accelerate

This library is proposed by Hugging Face. Hugging Face Accelerate is a library for simplifying and accelerating the training and inference of deep learning models. It provides an easy- to-use API that abstracts away much of the low-level details of distributed training and mixed- precision training. Some of the key concepts are:

- Distributed training
- Mixed-precision training
- Data parallelism

## Dataset

Even for the most basic model training, one should gather related and high-quality data. For fine-tuning a Stable Diffusion model, this is also important and a must. In order to get consistent and high-quality outputs, we need to collect high-quality images. The training images should match the expected output and resized to 512 x 512 in resolution.

Gathering images is not the only thing we need to be careful. Some artifacts can affect the output quality of our model. Artifacts such as motion blur or low resolution will affect the generated images. We need to be careful about any unwanted text, watermarks or icons in your training datasets.

Before preparing training images, there are two terms we need to know. These are **instance images** and **class images**. Instance images are the custom images that represents the specific concept for Dreambooth training. This type of images should be in high-quality. Class images are the regularization images for prior-preservation. These images prevent us from over-fitting. The generation of these images should be done using the base pre-trained model. 

Depending on the desired output type, we can have two different cases. **Object** and **Style.** Let’s discuss each case clearly.

### Object

In this case, basically we are using an object. This object can be a guitar or a toy. The object should be in the foreground and the background should be clear and normal. There should not be a transparent background. Using transparent background can lead to have a border around the object. You can use some variations while preparing the input images. These are:

- Camera angle
- Taking photos in different places (changing background)
- Pose of the object
- Different lighting

The number of the images can be 5 to 20. Sometimes cropping the image and focusing only the object may lead to better results. Collecting diverse set of object images improves the quality of the outputs.

### Style

In this case, we are using the styles of the images we like. It can be from your own art collections or movies, animations, TV series that share a consistent style. There is a big difference between object and style case. In object case, we are preparing the dataset focusing on the object. However, in the style case, we are focusing on the style not the object. 

To make the dataset better, there should not be an object that appears in the images more than one. This helps us to train the model for style case and our model will be able generate different characters of the same style.

## Training

In this section, we will explain the most important parameters. 

- **pretrained\_model\_name\_or\_path**: path to pre-trained model that will be used in fine- tuning
- **pretrained\_vae\_name\_or\_path**: path to pre-trained vae.
- **max\_train\_steps**: total number of training steps.
- **instance\_data\_dir**: data of instance images.
- **class\_data\_dir**: training data of class images.
- **instance\_prompt**: prompt with identifier specifying the instance.
- **class\_prompt**: prompt to specify images in the same class as provided instance images.
- **num\_class\_images**: minimal class images for prior preservation loss.
- **learning\_rate**: initial learning rate.
- **lr\_scheduler**: scheduler type to use.
- **lr\_warmup\_steps**: number of steps for the warmup in the lr scheduler.

In training, we can use a custom identifier that is unique. It is recommended to use a unique identifier. By unique, we mean that the identifier is not in the original Stable Diffusion’s datasets. The steps of training is explained in the notebook. You can follow it to fine tune a model.

## Inference
In this section, we will first explain the parameters used for generating images.

- **prompt**: concise problem statement.
- **negative prompt**: way to use Stable Diffusion in a way that allows the user to specify what he doesn’t want to see, without any extra input.
- **number of samples**: number of the output.
- **guidance scale**: controls how similar the generated image will be to the prompt.

Also, there are some limitations in the inference step. These can be listed like:

- Language drift
- Overfitting
- Preservation loss

Preservation loss means that the model may not generate rare context which may not be included in the model or it’s difficult for the model to generate. The inability to synthesize images of rarer or more complex topics as well as variable subject fidelity, which can result in hallucinogenic shifts and discontinuous qualities, are further limitations. Language drift means prompting can change context. Finally, overfitting means if the input context is close to the output, the results may be the same with the training data. The subject might not be assessed or might be combined with the context of the uploaded images if there aren’t enough input photos. This also occurs when a context for an odd generation is prompted.

In the provided notebook, the steps of the inference is represented.

## Discussion

In this section, we will discuss two main key points.

- Discuss the advantages and challenges of using Dreambooth for personalized image generation.
    - There are some challenges of Dreambooth. The first one is not being able to accurately generate the prompted context. Possible reason is difficulty in generating both the subject and concept together because of the low number of training images. The second one is the the difference between context and appearance. The output can change according to prompt frequently. The third one is the overfitting issue. When the prompt is similar to the original setting in which the subject was seen, we can see an output similar to the real image.
- Provide recommendations or best practices for using Dreambooth and fine-tuning Stable Diffusion effectively.
    - Around 20 - 24 photos is perfect.
    - You have to use an even number of photos or the training stops early. (This is been fixed.)
    - The photos should be cropped to 512x512.
    - The subject should be centered in frame.
    - No motion blur or low resolution.
    - No watermarks.
    - Samples with good consistency.
    - No transparent background for object case.

## References

1. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2021). High-Resolution Image Synthesis with Latent Diffusion Models. *ArXiv*. /abs/2112.10752
2. Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., & Aberman, K. (2022). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation. *ArXiv*. /abs/2208.12242


[1]: https://huggingface.co/spaces/nuwandaa/dreambooth-teddy-bear-demo/tree/main/dreambooth-concept
[2]: https://huggingface.co/spaces/nuwandaa/dreambooth-teddy-bear-demo
[3]: https://github.com/ShivamShrirao/diffusers
