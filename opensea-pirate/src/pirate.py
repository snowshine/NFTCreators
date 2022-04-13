import requests
import shutil
import boto3
import time
import json
import math
import os

from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

sqs_queue_url = ""

boto3.setup_default_session(profile_name='nf2')

class SeaportPirate:
    def __init__(self):
        # initialize all the shared variables
        self.job_details = {}

        # set necessary AWS clients
        self.s3_client = boto3.client('s3')
        self.sqs_client = boto3.client("sqs", region_name="us-east-2")
        self.dynamodb = boto3.resource('dynamodb', region_name="us-east-2")

        self.s3_location = None
        self.job_table = None
        self.receipt_handle = None

        self.stats = {
            "DownloadedData": 0,
            "AlreadyDownloadedData": 0,
            "DownloadedImages": 0,
            "AlreadyDownloadedImages": 0,
            "FailedImages": 0
        }
    def get_headers(self):
        return {"Accept": "application/json", "X-API-KEY": self.job_details["api_key"]}

    def retrieve_job(self):
        response = self.sqs_client.receive_message(
            QueueUrl=sqs_queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=10,
        )

        print(f"Number of messages received: {len(response.get('Messages', []))}")
        if (len(response.get('Messages', [])) == 0):
            raise ValueError("No jobs available.") 

        for message in response.get("Messages", []):
            self.receipt_handle = message['ReceiptHandle']

            message_body = message["Body"]
            message_json = json.loads(message_body)

            message_keys = ["collection", "api_key", "s3_location", "job_table"]
            for _key in message_keys:
                if (_key in message_json.keys()):
                    self.job_details[_key] = message_json[_key]
                else:
                    raise ValueError(f"Key (${_key}) not available in job payload!")

    def create_folders(self):
        if not os.path.exists('./images'):
            os.mkdir('./images')

        if not os.path.exists(f"./images/{self.job_details['collection']}"):
            os.mkdir(f"./images/{self.job_details['collection']}")

        if not os.path.exists(f"./images/{self.job_details['collection']}/image_data"):
            os.mkdir(f"./images/{self.job_details['collection']}/image_data")

        if not os.path.exists(f"./images/{self.job_details['collection']}/image_asset"):
            os.mkdir(f"./images/{self.job_details['collection']}/image_asset")


    def get_total_supply(self):
        colletion_url = f"https://api.opensea.io/api/v1/collection/{self.job_details['collection']}/stats"

        response = requests.request("GET", colletion_url, headers=self.get_headers())
        if response.status_code == 200:
            supply = response.json()["stats"]["total_supply"]
            
            return int(supply)
        elif response.status_code == 404:
            print(f"Collection not found. Verify collection slug and try again.")
            raise ValueError('Unable to locate specified collection.')
        else:
            print(f"Unable to retrieve collection supply count. {response}")

    def augment_image(self, formatted_number):
        return
        # adapted from https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
        print('Augmenting image...')
        img = load_img(f"./images/{self.job_details['collection']}/image_asset/{formatted_number}.png")
        data = img_to_array(img)
        samples = expand_dims(data, 0)
        datagen = ImageDataGenerator(rotation_range=90)
        it = datagen.flow(samples, batch_size=1)
        for i in range(2):
            batch = it.next()
            image = batch[0].astype('uint8')
            pyplot.imsave(f"./images/{self.job_details['collection']}/image_asset/{formatted_number}_{i}.png", image)

    def retrieve_assets(self, cursor=None):
        asset_url = f"https://api.opensea.io/api/v1/assets?collection={self.job_details['collection']}&order_direction=desc&limit=50&include_orders=false"
        if (cursor):
            asset_url += f"&cursor={cursor}"

        response = requests.request("GET", asset_url, headers=self.get_headers())

        if (response.status_code == 200):
            response_json = response.json()

            for asset in response_json["assets"]:
                formatted_number = f"{int(asset['token_id']):04d}"
                print(f"\n#{formatted_number}:")
                # Check if data for the NFT already exists, if it does, skip saving it
                if os.path.exists(f"./images/{self.job_details['collection']}/image_data/{formatted_number}.json"):
                    print(f"  Data  -> [\u2713] (Already Downloaded)")
                    self.stats["AlreadyDownloadedData"] += 1
                else:
                    # Take the JSON from the URL, and dump it to the respective file.
                    dfile = open(f"./images/{self.job_details['collection']}/image_data/{formatted_number}.json", "w+")
                    json.dump(asset, dfile, indent=3)
                    dfile.close()
                    print(f"  Data  -> [\u2713] (Successfully downloaded)")
                    self.stats["DownloadedData"] += 1
                
                # Check if image already exists, if it does, skip saving it
                if os.path.exists(f"./images/{self.job_details['collection']}/image_asset/{formatted_number}.png"):
                    print(f"  Image -> [\u2713] (Already Downloaded)")
                    self.stats["AlreadyDownloadedImages"] += 1
                else:
                    # Make the request to the URL to get the image
                    if not asset["image_thumbnail_url"] == None:
                        image_url = asset["image_thumbnail_url"]
                    else:
                        image_url = asset["image_url"]
                    
                    image = requests.get(image_url)

                    # If the URL returns status code "200 Successful", save the image into the "images" folder.
                    if image.status_code == 200:
                        file = open(f"./images/{self.job_details['collection']}/image_asset/{formatted_number}.png", "wb+")
                        file.write(image.content)
                        file.close()
                        print(f"  Image -> [\u2713] (Successfully downloaded)")
                        self.augment_image(formatted_number)
                        self.stats["DownloadedImages"] += 1
                    # If the URL returns a status code other than "200 Successful", alert the user and don't save the image
                    else:
                        print(f"  Image -> [!] (HTTP Status {image.status_code})")
                        self.stats["FailedImages"] += 1
                        continue

            return response_json["next"]
        
        else:
            print(f"Error in asset retrieval. Received: {response.status_code}")
            # give some backoff time then send back the same cursor to try again
            time.sleep(10)
            return cursor

    def upload_image_assets(self, retries = 0):
        try:
            asset_filename = f"{self.job_details['collection']}_archive"

            print(f"Compressing ${self.job_details['collection']} image assets...")
            shutil.make_archive(asset_filename, 'zip', f"./images/{self.job_details['collection']}/image_asset")

            print(f"Uploading ${self.job_details['collection']} images to S3...")
            self.s3_client.upload_file(asset_filename+'.zip', self.job_details['s3_location'], self.job_details['collection']+'/'+asset_filename+'.zip')
        except:
            if (retries <= 3):
                print(f"Retrying upload... attempt {retries + 1} of 3")
                self.upload_image_assets(retries=(retries + 1))
            else:
                raise RuntimeError('Upload of image assets failed.')

    def upload_metadata_assets(self, retries = 0):
        try:
            metadata_filename = f"{self.job_details['collection']}_metadata_archive"

            print(f"Compressing ${self.job_details['collection']} metadata assets...")
            shutil.make_archive(metadata_filename, 'zip', f"./images/{self.job_details['collection']}/image_data")

            print(f"Uploading ${self.job_details['collection']} metadata to S3...")
            self.s3_client.upload_file(metadata_filename+'.zip', self.job_details['s3_location'], self.job_details['collection']+'/'+metadata_filename+'.zip')
        except:
            if (retries <= 3):
                print(f"Retrying upload... attempt {retries + 1} of 3")
                self.upload_image_assets(retries=(retries + 1))
            else:
                raise RuntimeError('Upload of image metadata failed.')


    def upload_assets(self, meta_only = False, retries = 0):
        self.upload_image_assets()
        self.upload_metadata_assets()

    def resolve_job(self):
        response = self.sqs_client.delete_message(
            QueueUrl=sqs_queue_url,
            ReceiptHandle=self.receipt_handle,
        )

    def publish_job_record(self):
        asset_filename = f"{self.job_details['collection']}_archive"
        metadata_filename = f"{self.job_details['collection']}_metadata_archive"

        table = self.dynamodb.Table(self.job_details['job_table'])
        response = table.put_item(
        Item={
                'collection_slug': self.job_details['collection'],
                'image_location': self.job_details['collection']+'/'+asset_filename+'.zip',
                'metadata_location': self.job_details['collection']+'/'+metadata_filename+'.zip',
                'model_location': '',
            }
        )
        return response

    def print_stats(self, supply):
        print(f"""
        Finished downloading collection.

        Statistics
        -=-=-=-=-=-

        Total of {supply} units in collection "{self.job_details['collection']}".

        Downloads:

        JSON Files ->
            {self.stats["DownloadedData"]} successfully downloaded
            {self.stats["AlreadyDownloadedData"]} already downloaded

        Images ->
            {self.stats["DownloadedImages"]} successfully downloaded
            {self.stats["AlreadyDownloadedImages"]} already downloaded
            {self.stats["FailedImages"]} failed""")

    def main(self):
        try:
            self.retrieve_job()
            self.create_folders()

            supply = self.get_total_supply()
            print(f"Total Supply: {supply}")

            min_calls = math.ceil(supply / 50)
            cursor = None
            for _ in range(min_calls):
                start_request = time.time()
                cursor = self.retrieve_assets(cursor)
                end_request = time.time()
                delta = end_request - start_request
                if (delta < 5):
                    time.sleep(5 - delta)
            #self.upload_assets()
            self.resolve_job()
            #self.publish_job_record()

            self.print_stats(supply)
        except Exception as e:
            print(e)
            print(f"Encountered error. See above for details.")

if __name__ == "__main__":
    SeaportPirate().main()
