# Token Inference
The InferenceDaemon is meant to be a long-running process to support asynchronous generation of model output.

### Code Path Description
1. The daemon long-polls an Amazon SQS queue for inferece requests made by a user
2. If a request is found the daemon updates the job state in the state table
3. The daemon checks if the require model is locally available
4. If model is not locally available, the model is downloaded from S3
5. The daemon generates an output with the local model file
6. The daemon uploads the generated image to S3
7. The job is marked as complete and the S3 image location is added to the job state

### Required Infrastructure
1. An Amazon S3 bucket to house model files
    * The bucket should have directories named after collection slugs
    * Models should be names `{collection_slug}_generator.zip` 
2. An Amazon SQS queue that holds the inference requests
    * Required fields for SQS messages are
        * CollectionSlug [string]
        * JobId [string]
3. An Amazon DynamoDB table to hold data about each available collection with reference to the model location in S3
    * Table schema
        * (PK) collection_slug [string]
        * asset_count [number]
        * model_location [string]
4. An Amazon DynamoDB table to hold state machine data about inference requests
    * Table schema
        * (PK) JobId [string]
        * (SK) CollectionSlug [string]
        * AssetLocation [string]
        * Status [string]

### Running Inference Daemon
Below is a sample bash script that can be used to start the daemon. Add the infrastructure names as noted.

```
export MODEL_BUCKET_NAME=""
export SQS_QUEUE_URL=""
export DYNAMO_ASSETS_TABLE_NAME=""
export DYNAMO_JOBS_TABLE_NAME=""

python3 ./InferenceDaemon.py
```
