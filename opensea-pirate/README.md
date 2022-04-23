# OpenSea Pirate
The main code is adapted from [OpenSea-NFT-Stealer](https://github.com/sj-dan/OpenSea-NFT-Stealer) though heavily modified to account for updates to the OpenSea API.

### Overview
The primary goal of this package is to extract the metadata and image assets of an NFT asset collection via the OpenSea API.

### Code Path Description
This is not meant to be run as a long running process. It works best when executed locally to collect a single dataset as it make take upwards of 60 minutes to complete processing an average sized collection of 10K assets.

### Required Infrastructure
1. An Amazon S3 bucket to house model files
    * The bucket should have directories named after collection slugs
        * Models should be names {collection_slug}_generator.zip
2. An Amazon SQS queue that holds the inference requests
    * Required fields for SQS messages are
        * collection [string]
        * api_key [string]
        * s3_location [string]
        * job_table [string]
3. An Amazon DynamoDB table to hold data about each available collection with reference to the model location in S3
    * Table schema
        * (PK) collection_slug [string]
        * asset_count [number]
        * image_location [string]
        * metadata_location [string]

### Running Opensea Pirate
 First, an SQS message needs to be created that holds the data necessary for the collection. This is required to specify all parameters needed by the collection process.
```
    sqs_client = boto3.client("sqs")

    message = {"collection": "COLLECTION_SLUG","api_key": "YOUR_API_KEY", "s3_location": "BUCKET_NAME", "job_table": "TABLE_NAME"}
    response = sqs_client.send_message(
        QueueUrl="SQS_QUEUE_URL",
        MessageBody=json.dumps(message)
    )
```

Once a collection definition has been submitted to SQS, the collection process just needs to be executed. This process may take greater than 60 minutes to complete.
```
python3 pirate.py
```


