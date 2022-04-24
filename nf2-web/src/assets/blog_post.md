# Non-Fungible Token Fabrication with Generative Adversarial Networks
###### Allen Chang, Jason Coffman, Cindy Xie

## Overview
Non-Fungible Token are, according to the Ethereum website, a way to represent anything unique as an Ethereum-based asset<sup>[1]</sup>. A rapidly growing use of the NFT technology that rose to prominence in the last couple years is the minting and transaction of image based assets listed via NFT marketplaces. These image based assets are often offered by their original artists in collections of thousands of unique, yet similarly styled, images with slight variations. With the rapid influx in NFT supply, and a growing interest from the public in decentralized currency the NFT market has quickly ballooned into a multi-million dollar industry. The total market value of NFT assets minted on the Ethereum blockchain and listed on the OpenSea marketplace is estimated at $840 million USD as of March 2022, with significant growth starting in early-2021<sup>[2]</sup>.

While other researchers have examined the use of Generative Adversarial Networks (GANs) as a style transfer mechanism for both Anime<sup>[3]</sup> and cartoons<sup>[4]</sup> styles, there has not been notable work done in the domain of NFTs, which maintain their own unique art styles. There are instances of computer generated collections on the OpenSea market place, such as the [Computer Generated Ape Club](https://opensea.io/collection/computer-generated-ape-club), but their methods and techniques are not made public. For our capstone project, we had multiple goals for our project. First, we wanted to an opportunity to apply deep learning in GANs. GANs has taken off in popularity with several applications, which we touched on as a topic in our MADS Deep Learning class. The Capstone provided an opportunity to apply it and build optimized models that generate GAN images. The second motivation is to dive into the Non-Fungible Token (NFT) and the underlying blockchain domain. NFTs have become a hot investment topic as there have been tokens, such as the CryptoPunks #5822 sold for $23.7M in February of 2022. Part of our goal was analyzing the metadata information that each NFT collection possess to better understand how it influences the total price thru exploratory data analysis techniques with data collected from the OpenSea API. By identifying the subset of tokens in a collection that have higher total price value, we created a GAN model that has the potential to generate GANs with higher values than ones generated from the whole collection. Our final goal was to build an end-to-end web application that enabled users to view a collection of GAN models that a user can select for generating an NFT token. In summary, out team strived to learn a new data science domain, apply GAN techniques, and develop an end user application that tied all the elements together to produce novel GANs of interest.

## Project Data

The single data source utilized for this project was the OpenSea API. While there are other NFT marketplaces, OpenSea was chosen as it is the top NFT marketplace by trade volume according to [Dapp Radar](https://dappradar.com/nft/marketplaces). Since one of the stated goals was to investigate NFT collections with high valuations, it was important that the selected data source contain relevant, recent transaction information in addition to containing references to the original NFT image asset for retrieval.

### Data Collection
An API key is required to perform high volume queries on the [OpenSea API](https://docs.opensea.io/reference/api-overview) and, as such, one was provisioned for this project.

A typical NFT collection contains 10,000 unique image assets, each with a set of unique traits and associated transaction details. Given the size and volume of tokens needed, an automated collection library was written to handle this process. The library relies on an Amazon Simple Queue Service message to launch a collection job; this message contains the NFT collection slug, a unique string that identifies an NFT collection, the API key that will be used with the OpenSea API, an Amazon DynamoDB table name to store information about the collect, and an Amazon S3 bucket to upload the image assets and metadata to. The library created to retrieve the data sets can be found on [GitHub](https://github.com/snowshine/NFTCreators/tree/main/opensea-pirate).

Below is an overview of the data that is available in an NFT asset metadata file; a full description of the data model can be found in the [OpenSea API documentation](https://docs.opensea.io/reference/asset-object).

##### Table 1: OpenSea Asset metadata schema
| Key                | Value Type      |
| :---               |    ---:         |
| token_id           | number          |
| image_url          | string          |
| background_color   | string          |
| name               | string          |
| external_link      | string          |
| asset_contract     | contract object |
| owner              | string          |
| traits             | traits object   |
| last_sale          | string          |


At the beginning of the collection process OpenSea utilized an `offset` based pagination system which capped asset collection at 10,000 NFTs. However, in mid-March 2022 OpenSea updated their API to utilize a `cursor` pagination system that had no limit on the number of NFTs that could be retrieved. While this was beneficial, as it enabled the collection of larger NFT collections, it required significant reworking of our data collection process.

### Data Storage and Retrieval
We utilized Amazon S3 as a form of shared file system. Compressing the image and metadata files and storting them in Amazon S3 enabled rapid retrieval of the files needed for particular workflows. For example, when training a model for a specific NFT collection, only a single collection ZIP file needs to be downloaded. The schema for our file system was as follows:
```
nft-assets / 
    ↳ {collection_slug} / 
        ↳ {collection_slug}_archive.zip
        ↳ {collection_slug}_metadata_archive.zip
        ↳ {collection_slug}_generator.zip
```

Maintaining a simple, hierarchical structure based off collection slugs enabled any process performed to be fully automated in theory. Given any particular collection slug, assuming it exists, the data and models for the slug can be simply extracted for use.

## Metadata Analysis
For each NFT within a collection, a metadata file compliments the token retrieved from the OpenSea API. Metadata is provided in json format and provides informative details associated with the token. 
Some of the key attributes include:
 - ID – identifier
 - Name: name of NFT
 - Image_url: url to token to universally unique location
 - Asset_ contract – collection of contract details, (eg. Asset_contract_type, created_date)
 - Owner – details of the current owner (e.g., username, profile_img_url)
 - Creator – details of the token creator
 - Traits – collection of traits including type, value, count, order, display_type
 - Last_sale – collection of details associated with last sale including transaction, payment token and total_price

One of the goals of the project included exploratory data analysis of the metadata for all the collections to better understand the details of blockchain transactions/elements and the features that potentially drive the total price of NFTs. I spent a good amount of time getting up to speed on NFT/Blockchain and the ERC-721 standard behind the smart contract implementation that defines the ownership details, security, and metadata. In understanding those elements, I searched for attributes that can influence the total price in a meaningful way. Given that, we narrowed down to the traits and last_sale collections within the metadata in search of correlations that can influence the subset of NFTs chosen as input to the GAN training, with the goal of generating monetarily valuable NFTs. In learning about NFT traits, the combinations of traits including rare traits often influence prices, although there are other aspects that can influence prices (such as owners’ popularity, etc.) that were not included in the analysis, so we were cautious of potential confounders. Nevertheless, we continued analyzing the traits and trait combinations. For example, the following shows the top traits and most common traits of the Bored Ape Yacht Club collection.

![bored-ape-rare-trait.svg](/assets/bored-ape-rare-trait.svg)
##### Figure 1: Histogram of the top 20 most rare traits in the BoredApeYachtClub NFT collection

![bored-ape-common-trait.svg](/assets/bored-ape-common-trait.svg)
##### Figure 2: Histogram of the top 20 most common traits in the BoredApeYachtClub NFT collection

The output is what I would have expected, with the rare traits being a more eclectic mix such as the top one of bored unshaven pizza. The traits and the counts are interesting in themselves, but to be useful we needed a way to capture them in a quantitative feature. We followed the approach utilized by a popular NFT rarity ranking web site [rarity.tools](https://rarity.tools/). We implemented the same calculation to develop a score feature. The approach followed was adopted from [Ranking Rarity: Understanding Rarity Calculation Methods](
https://raritytools.medium.com/ranking-rarity-understanding-rarity-calculation-methods-86ceaeb9b98c). After creating a rarity score for each, I created a scatter plot of the rarity score against the total price translated from Eth to USD in log form due to variability to see what the relationships looked like. For example, as you can see with the Bored Ape Collection, it is quite a spread.

![bored-ape-rarity-scatter.svg](/assets/bored-ape-rarity-scatter.svg)
##### Figure 3: Rarity score against log10 of total sale price for tokens in the BoredApeYachtClub NFT collection

We also visualized the total price distribution for each collection in histogram plots. For example, the following is the distribution for the Bored Ape Collection.

![bored-ape-total-price.svg](/assets/bored-ape-total-price.svg)
##### Figure 4: Histogram of log10 of the token sale price (USD) for the BoredApeYachtClub NFT collection

We developed these visualizations along with many more for all the collections, which can be viewed in the Collections_Metadata notebook on our project Github page. Given the variability and inconclusiveness of the relationship between our traits’ rarity score, we decided to utilize the total_cost as the feature to create a “highest price filter” collection and model from the top 15 percent from the Bored Ape Collection. Helper functions were created in the Collections_Metadata_Processing notebook for creating a subset of tokens/metadata based on rarity score or total cost. This enabled us to create a “tuned” version of the Bored Ape Collection, which was one of the project goals. The EDA exercise also created summary statistics (ex: mean and median of total price) that is being utilized in the web application for context and gives the user insight in what potentially to expect from the GAN chosen for token generation. 
The final plot below is a strip plot of the total cost distribution across all the collections analyzed.

![price-by-collection.svg](/assets/price-by-collection.svg)
##### Figure 5: Plot of log10 of the token sale price (USD) for all analyzed collections

As you can see from the distribution across the five collections analyzed, the price distribution is consistent (on a log scale). These collections represent the source of input in training the GANs that are created and hosted for inference in our web application.

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
    - Using Strided Convolutions for downsampling and upsampling instead of pooling
    - Adding Gaussian Weight Initial noise to every layer of generator
    - Using Batch Normalization
    - Using LeakyReLU instead of ReLU, with a default slope of 0.2
    - Using Dropout in discriminator
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

The training starts with image processing which preparing image data as Tensor data set and then start the training loop. We save the model as well as the generated images every 15 epochs, so we can view and monitor the training progress. It also helps when the training is interrupted due to whatever reason such as session expiring, timeout, etc.

![gan_train_flow.JPG](/assets/gan_train_flow.jpeg)
##### Figure 6: High level diagram of project flow

GAN is hard to train with some well-known challenges like non-convergence. Since GAN has 2 models and each has its own loss function and measured individually, but there is no overall objective loss function, so we can't determine the relative/absolute GAN model quality from loss alone, and we don't know when to stop training. For this project, we ran about 500 to 1000 epochs for every collection. Training for a long time doesn’t always make the generator better, it can be worse. As you can see from below training examples, the generator seems lost direction around epoch 600, but eventually get back up again after continuous training.

![gan_training.jpg](/assets/gan_training.jpeg)
##### Figure 7: Overview of BoredApeYachtClub training epochs

### GAN Evaluations

Due to the challenge of non-convergence and lack of overall loss function, there is no objective ways to assess the model performance. We mostly rely on subjective evaluation or manual inspections. 

The common understanding of a GAN model is “good” when an equilibrium is reached between generator and discriminator, typically when the discriminator’s loss is around 0.5. We have saved both generator and discriminator's loss for each epoch and chart them after training. A sample chart is shown below which is an example that the generator lost its patience to be a good learner after 700 epochs. Loss charts give us some clues, but can’t determine the performance.

![training_loss_chart.jpg](/assets/training_loss_chart.jpeg)
##### Figure 8: Training loss for BoredApeYachtClub training over 800+ epochs

We also tried to evaluate our model with the two most popular quantitative measures for GAN: Inception Score (IS) and Frechet Inception Distance (FID).

Inception Score (IS) use the pre-trained Inception v3 model to classify the generated images. The score has 2 criteria for the generated images:
- quality: how much each image looks like a known class
- variety: how diverse the set represent different known classes.

A higher inception score indicates better-quality generated images, but we got very poor IS scores. For almost every collection we trained, we have IS score between 0.9 to 1, not just for the generated images, but also for the original NFT images. Considering NFTs are new art, not like anything else in the known class for Inception v3 model, we think IS score is just not applicable to us.

Same as the inception score, the Frechet Inception Distance (FID) score also uses the inception v3 model, but the score calculation method is different. The score uses both generated images and original images, and calculate the distance between real images and generated images using the Frechet distance. A lower FID score, with 0 as the best score, indicates more realistic images that match the statistical properties of real images. The common agreement is FID is consistent with human judgments and is more robust to noise than IS. Comparing to IS, our FID seems very good, normally range from 0.05 to 0, kind of hard to believe. From our evaluation result (as shown in the example below), sometimes the FID do reflect human judgments, but there are times it's out of the place.

![gan_evaluation.jpg](/assets/gan_evaluation.jpeg)
##### Figure 9: Selected outputs from training epochs of BoredApeYachtClub and Azuki collections

## Web Application
The purpose of the NFT Web Application (Non-Fungible Fakes) is to enable users to explore the results of the metadata analyses performed and interact with the models produced by this project. This was achieved by building a platform for users to explore details about specific NFT collections and generate image assets that they may download. 

### Token Generation and Analytics
A primary goal of the web application was to enable users to directly interact with the trained GAN models to generate novel image assets. The images are created via a long-running process hosted on an Amazon EC2 instance, and then the images are put into an Amazon S3 bucket. While the process designed is not highly efficient, with moderate throughput it produces new tokens in less than 60 seconds. A high-level architecture of the token generation process can be found in _Figure 10_.

![token generation infrastructure](/assets/generate.svg)
##### Figure 10: Architecture diagram of token generation process

In addition to providing a view that allows users to generate new tokens, the visualization and statistics provided enable users to conduct high-level discovery of the collection they are viewing. The visualization and statistics shown on an NFT collection's page provide insights into the sales, traits and rarity of the tokens within collection. The asset sales histogram, for example, provides a simple summary of the distribution of sale prices of tokens.

### Disclaimer
Some of the data visualizations presented within the web application do not adhere to strict best practices. This was done in the best interest of aesthetics and given the constraints of the charting library utilized. While these visualizations do not follow best practices, they are based off of visualizations described in the above section (Metadata Analysis) which were created in the spirit of following visualization best practices.

## Ethical Considerations
For our project, there are ethical issues to be mindful and careful about. In analyzing the NFT collections that served as our source, one can see that the differentiator is the combination of unique traits. Some of the traits can be shocking, exaggerated, and outlandish which can sometimes be alluring to collectors. The issue is that those traits can perpetuate negative stereotypes that can cause unintended harm to the target demographic. Since we are using NFT input across thousands of samples, it is difficult to catch those that our GAN may generate. To that end, we as Data Scientist need to be mindful of those unintended consequences in the tokens our models generate and remove or refine those models if possible or remove them as needed.

## Summary
This project delved deeply into the Non-Fungible Token space and worked to examine the current state of the image-based NFT asset landscape. Our final output, the Non-Fungible Fakes web application, enables users to gain insights into NFT collections via metadata visualization and statistics, and empowers them to utilize the generator models that were produced. The intent behind all of this is to expose users to the concept and power of GANs, provide them a glimpse into the value, and what drives the value, of NFT assets, and allow them to take away an image asset based on these collections for personal use.

## Statement of Work
Cindy works with the Generative Adversarial Networks (GANs) models including building and tunning the models. She runs the training for all the NFT collections, evaluates the result after training, and select the generators for the web application to use.

Jason handled the collection and organization of the data that was used, which included creating re-usable Python code to paginate and retrieve all image assets, download and compress them, and upload the compressed file to Amazon S3. He built and hosted the web application for the project using Firebase for hosting and Amazon S3 to power much of the web application logic. Jason also wrote a Python daemon to run model inference as described in the _Token Generation and Analytics_ section, powering the web application's image generation feature.

Allen performed the metadata Exploratory Data Analysis tasks, which included the data engineering processing of json files to merged set of pandas DataFrames that supported visualization of various metadata elements within a collection and across collections. From that, new candidate features were explored, considered, and created to support filtering on monetarily valuable NFTs within a collection. He implemented helper functions that supported the creation of new input collections based on features of interest (total price, rarity score), which was used to feed the model generation process. Allen also created visualizations and summary metrics of the collections that served as the source for the metadata context provided in the Web Application UI.

The team collectively contributed to the stand ups, status, final report blog and video deliverables for the project.

## References
<sup>[1]</sup>Non-fungible tokens (NFT). ethereum.org. (n.d.). Retrieved April 24, 2022, from https://ethereum.org/en/nft/ 

<sup>[2]</sup>Published by Statista Research Department, &amp; 5, A. (2022, April 5). Art blocks: NFT market cap 2022. Statista. Retrieved April 24, 2022, from [https://www.statista.com/statistics/1291885/](https://www.statista.com/statistics/1291885/art-blocks-nft-market-cap/#:~:text=Market%20cap%20of%20Art%20Blocks%20NFT%20projects%20worldwide%20November%202020%2DMarch%202022&amp;text=As%20of%20March%2031%2C%202022,platform%20focusing%20on%20generative%20art). 

<sup>[3]</sup>Li, B., Zhu, Y., Wang, Y., Lin, C.-W., Ghanem, B., &amp; Shen, L. (2021). Anigan: Style-guided generative adversarial networks for unsupervised anime face generation. IEEE Transactions on Multimedia, 1–1. https://doi.org/10.1109/tmm.2021.3113786 

<sup>[4]</sup>Shu, Y., Yi, R., Xia, M., Ye, Z., Zhao, W., Chen, Y., Lai, Y.-K., &amp; Liu, Y.-J. (2021). Gan-based multi-style photo cartoonization. IEEE Transactions on Visualization and Computer Graphics, 1–1. https://doi.org/10.1109/tvcg.2021.3067201 