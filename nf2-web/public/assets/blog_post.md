## Project Data
### Data Source
The single data source utlizied for this project was the OpenSea API. While there are other NFT marketplaces, OpenSea was chosen as it is the top NFT markplace by trade volume according to [Dapp Radar](https://dappradar.com/nft/marketplaces). Since one of the stated goals was to investigate NFT collections with high valuations, it was important that the selected data source contain relevant, recent transaction information.

### Data Collection
An API key is required to perform high volume queries on the [OpenSea API](https://docs.opensea.io/reference/api-overview) and, as such, one was provisioned for this project. 

The average NFT collection contains 10,000 unique image assets and an associated metadata file. Given the size and volume of tokens needed, an automated collection library was written to handle this process. The library relies on an Amazon Simple Queue Service message to launch a collection job; this message contains the NFT collection slug, a unique string that identifies an NFT collection, the API key that will be used, an Amazon DynamoDB table to store information about the collect, and an Amazon S3 bucket to upload the image assets and metadata to. The library created to retrieve the datasets can be found on [GitHub](https://github.com/snowshine/NFTCreators/tree/main/opensea-pirate).

Below is an overview of the data that is available in an NFT asset metadata file; a full description of the data model can be found in the [OpenSea API documentation](https://docs.opensea.io/reference/asset-object).

#### Asset metadata schema
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


At the beginning of the collection process OpenSea utilizied an `offset` based pagination system which capped asset collection at 10,000 NFTs. However, in mid-March 2022 they updated their API to utilize a `cursor` pagination system that had no limit on the number of NFTs that could be retrieved. While this was beneficial as it enabled the collection of larger NFT collections, it required significant reworking of our data collection process.

### Data Storage and Retrieval
We utilized Amazon S3 as a form of shared file system across the team. The schema for our file system was as follows:
```
nft-assets / 
    ↳ {collection_slug} / 
        ↳ {collection_slug}_archive.zip
        ↳ {collection_slug}_metadata_archive.zip
```

## NFT Web Application
### Purpose
The purpose of the NFT Web Application (Non-Fungible Fakes) is to enable users to explore the analyses performed and interact with the models produced by this project. The aim was that it would produce both an informative and educational experience for users.

### Collection Analytics

### Disclaimer
Some of the data visualizations presented within the web application do not adhere to strict best practices. This was done in the best interest of aesthetics and given the contraints of the charting library utilized. While these visualizations do not follow best practices, they are based off of visualzations above which were created in the spirit of following visualization best practices.