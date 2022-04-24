## Modeling

### Generative Adversarial Networks (GANs)
Does computer have imagination to create new arts? GAN seems promising. 

Generative Adversarial Networks (GANs) are a rapidly developing generative deep learning model which has the ability to generate photorealistic examples of objects or people. It's an unsupervised learning model that involves automatically discovering and learning the regularities or patterns in input data, and then use these learned patterns to generate or output new examples that are similar to the original dataset.

One main goal of this project is building a GAN model and using NFT collections as training data, so the GAN can produce new tokens with the same characteristics as the training NFT collections. The generated new tokens, as a product of artificial creativity, can inspire a human artist or have its own value.

A GAN model composes two neural network models:
- the artist - a generator model to generate new examples
- the critics - a discriminator model that classify input as either real or fake. 

The two models are trained together with competing goals, until they are reaching a dynamic balance that discriminator model is fooled about half the time, meaning the generator model is generating plausible examples.

There are various GAN models and styles, our GANs are based on [Deep Convolutional GAN (DCGAN)](https://www.tensorflow.org/tutorials/generative/dcgan) which works better for image data as in our case. We did brief research of BigGAN - a latest and state-of-the-art model in the GAN family that capable of generating both high-resolution and high-quality images. But BigGAN utilizes techniques such as larger batch size which is a challenge to us due to our computing environment constraints.

### DCGAN Model and Best Practices
We started to build the GAN with small 32x32 RGB NFT images, but soon find out the result is better with higher resolution training data. So, we have 2 GAN models, one is supporting 32x32x3 images, and the other, which is used to train
most of our collections, is the 128x128x3 GAN model.

A plot of our 128X128 GAN model can be viewed here:
- [generator](https://github.com/snowshine/NFTCreators/blob/main/docs/generator_plot.png)
- [discriminator](https://github.com/snowshine/NFTCreators/blob/main/docs/discriminator_plot.png)

We tuned our model with most of the DCGAN best practice and found most of them are beneficial to the training outcome, though some may need more experiments and tweaking before they start showing improvements. 

Following is a list of best practice we followed:
- Generator/Discriminator:
    - Using Strided Convolutions for Downsample and Upsample instead of pooling
    - Adding Gaussian Weight Initial noise to every layer of generator
    - Using Batch Normalization
    - Using LeakyReLU instead of ReLU, with a default slope of 0.2
    - Using Dropout in descriminator
    - Using the tanh activation function to ensure generator output range [-1,1]
- Optimizer:
    - Using the Adam version of stochastic gradient descent (SGD)
    - Using Two Time-Scale Update Rule (TTUR): different learning rates for generator and discriminator
    - Tuning learning rate: start with 0.0002; Using beta1 momentum of 0.5 instead of the default of 0.9
- Training Process:
    - Scale/normalize images from [0,255] to the range of [-1,1]    
    - Train the discriminator with separate batches of real and fake images
    - Using one-sided Label Smoothing to tune down discriminator to avoid overconfidence

### GAN Training

We trained around 10 NFT collections. Half of all the collections is 10K NFT collections, the other half are 15k, 20k, 33k and 90k. In general, we found larger dataset produces better result which is reasonable. 

The training starts with image processing which preparing image data as Tensor dataset and then start the training loop. We save the model as well as the generated images every 15 epochs, so we can view and monitor the training progress. It also helps when the training is interrupted due to whatever reason such as session expiring, timeout, etc.

<img src="https://github.com/snowshine/NFTCreators/blob/main/docs/gan_train_flow.JPG">

GAN is hard to train with some well-known challenges like non-convergence. Since GAN has 2 models and each has its own loss function and measured individually, but there is no overall objective loss function, so we can't determine the relative/absolute GAN model quality from loss alone, and we don't know when to stop training. For this project, we ran about 500 to 1000 epochs for every collection. Training for a long time doesn’t always make the generator better, it can be worse. As you can see from below training examples, the generator seems lost direction around epoch 600, but eventually get back up again after continuous training.

<img src="https://github.com/snowshine/NFTCreators/blob/main/docs/gan_training.jpg">

### GAN Evaluations

Due to the challenge of non-convergence and lack of overall loss function, there is no objective ways to assess the model performance. We mostly rely on subjective evaluation or manual inspections. 

The common understanding of a GAN model is “good” when an equilibrium is reached between generator and discriminator, typically when the discriminator’s loss is around 0.5. We have saved both generator and discriminator's loss for each epoch and chart them after training. A sample chart is shown below which is an example that the generator lost its patient to be a good learner after 700 epochs. Loss charts give us some clues, but can’t determine the performance.

<img src="https://github.com/snowshine/NFTCreators/blob/main/docs/training_loss_chart.JPG">

We also tried to evaluate our model with the two most popular quantitative measures for GAN: Inception Score (IS) and Frechet Inception Distance (FID).

Inception Score (IS) use the pre-trained Inception v3 model to classify the generated images. The score has 2 criteria for the generated images:
- quality: how much each image looks like a known class
- variety: how diverse the set represent different known classes.

A higher inception score indicates better-quality generated images, but we got very poor IS scores. For almost every collection we trained, we have IS score between 0.9 to 1, not just for the generated images, but also for the original NFT images. Considering NFTs are new art, not like anything else in the known class for Inception v3 model, we think IS score is just not applicable to us.

Same as the inception score, the Frechet Inception Distance (FID) score also uses the inception v3 model, but the score calculation method is different. The score uses both generated images and original images, and calculate the distance between real images and generated images using the Frechet distance. A lower FID score, with 0 as the best score, indicates more realistic images that match the statistical properties of real images. The common agreement is FID is consistent with human judgments and is more robust to noise than IS. Comparing to IS, our FID seems very good, normally range from 0.05 to 0, kind of hard to believe. From our evaluation result (as shown in the example below), sometimes the FID do reflect human judgments, but there are times it's out of the place.

<img src="https://github.com/snowshine/NFTCreators/blob/main/docs/gan_evaluation.JPG">



## Statement of Work

Cindy works with the Generative Adversarial Networks (GANs) models including  building and tunning the models. She runs the training for all the NFT collections, evaluates the result after training, and select the generators for the web application to use.

