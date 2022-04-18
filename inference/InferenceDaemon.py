import os
import time
import json
import boto3
import utils
import zipfile
import traceback

from enum import Enum
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt

boto3.setup_default_session(profile_name='nf2')

MODEL_BUCKET_NAME = os.environ.get('MODEL_BUCKET_NAME')
SQS_QUEUE_URL = os.environ.get('SQS_QUEUE_URL')
DYNAMO_JOBS_TABLE_NAME = os.environ.get('DYNAMO_JOBS_TABLE_NAME')
DYNAMO_ASSETS_TABLE_NAME = os.environ.get('DYNAMO_ASSETS_TABLE_NAME')

class JobStatus(Enum): 
    """
        NOT_STARTED: Job created, not accepted
        PENDING: Job accepted, pending validation
        PROCESSING: job validated, inference begun
        COMPLETE: inference complete, output uploaded
        FAILED: some step failed
    """
    NOT_STARTED = 'NOT_STARTED'
    PENDING = 'PENDING'
    PROCESSING = 'PROCESSING'
    COMPLETE = 'COMPLETE'
    FAILED = 'FAILED'

class InferenceDaemon:
    def __init__(self):
        self.dynamodb = boto3.resource('dynamodb', region_name='us-east-2')
        self.sqs = boto3.client('sqs', region_name='us-east-2')
        self.s3 = boto3.client('s3')

    def update_inference(self, jobId, slug, status, location=None):
        table = self.dynamodb.Table(DYNAMO_JOBS_TABLE_NAME)
        update_payload = {
            'JobId': jobId,
            'CollectionSlug': slug,
            'Status': status,
        }

        if (not(location is None)):
            update_payload['AssetLocation'] = location

        table.put_item(
            Item=update_payload
        )

    def poll_messages(self):
        """
        payload:
            - jobId<string>: UUID created for requested generation
            - collectionSlug<string>: Slug for requested generation
        """
        queue_url = SQS_QUEUE_URL

        # Long poll for message on provided SQS queue
        try:
            response = self.sqs.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20
            )

            messages = response.get('Messages', [])
            if (len(messages) == 0):
                raise ValueError("No jobs available.") 

            return messages
        except Exception as e:
            print(e)
            return None

    def retrieve_remote_model(self, slug):
        table = self.dynamodb.Table(DYNAMO_ASSETS_TABLE_NAME)
        response = table.get_item(Key={'collection_slug': slug})  
        model_location = response['Item']['model_location']
        
        utils.create_model_path_from_slug(slug)
        local_zip_location = utils.get_model_path_from_slug(slug)
        local_model_location = utils.get_model_directory_from_slug(slug)

        self.s3.download_file(MODEL_BUCKET_NAME, model_location, local_zip_location)

        with zipfile.ZipFile(local_zip_location, 'r') as zip_ref:
            zip_ref.extractall(local_model_location)

    def retrieve_model(self, slug):
        if(not(utils.model_does_exist(slug))):
            self.retrieve_remote_model(slug)
        
        local_model_location = utils.get_model_to_load_from_slug(slug)

        return keras.models.load_model(local_model_location)


    def generate_fake(self, model, SEED_DIM=100):
        noise = tf.random.normal([1, SEED_DIM])
        generated_image = model(noise, training=False)
        
        return generated_image[0,:,:,0].numpy()

    def upload_fake(self, source, destination):
        self.s3.upload_file(source, MODEL_BUCKET_NAME, destination)

    def clear_message(self, receipt_handle):
        response = self.sqs.delete_message(
            QueueUrl=SQS_QUEUE_URL,
            ReceiptHandle=receipt_handle,
        )

    def process_message(self, message):
        body = json.loads(message["Body"])
        slug = body["CollectionSlug"]
        jobId = body["JobId"]

        try:
            self.update_inference(jobId, slug, JobStatus.PENDING.value)
            model = self.retrieve_model(slug)

            self.update_inference(jobId, slug, JobStatus.PROCESSING.value)
            image = self.generate_fake(model)

            utils.create_output_image_path()
            local_destination = f'./tmp/{jobId}.jpeg'
            remote_desination = f'generated/{slug}/{jobId}.jpeg'

            plt.imsave(local_destination, image)

            self.upload_fake(local_destination, remote_desination)
            self.update_inference(jobId, slug, JobStatus.COMPLETE.value, remote_desination)

        except Exception as e:
            print(e)
            print(traceback.format_exc())
            self.update_inference(jobId, slug, JobStatus.FAILED.value)

        finally:
            self.clear_message(message['ReceiptHandle'])
    
    def sleep(self):
        time.sleep(10)

    def main(self):
        while True:
            print('Attempting to retrieve messages...')
            messages = self.poll_messages()
            if (messages is None):
                self.sleep()
                continue

            for message in messages:
                self.process_message(message)

            self.sleep()

if __name__ == "__main__":
    if (MODEL_BUCKET_NAME is None):
        print('No bucket name specified.')
        exit(1)
    if (SQS_QUEUE_URL is None):
        print('No queue URL specified.')
        exit(1)
    if (DYNAMO_ASSETS_TABLE_NAME is None):
        print('No assets table specified.')
        exit(1)
    if (DYNAMO_JOBS_TABLE_NAME is None):
        print('No jobs table specified.')
        exit(1)

    InferenceDaemon().main()