import requests
import shutil
import json
import math
import os

collection = "adidasoriginals"
# TODO: Replace with your API Key - but never upload the key!
api_key = "USER_PROVIDED_API_KEY"
s3_location = "nf2-assets"

class SeaportPirate:
    def __init__(self, collection_slug, api_key):
        self.collection_slug = collection_slug
        self.api_key = api_key
        self.headers = {"Accept": "application/json", "X-API-KEY": self.api_key}
        self.stats = {
            "DownloadedData": 0,
            "AlreadyDownloadedData": 0,
            "DownloadedImages": 0,
            "AlreadyDownloadedImages": 0,
            "FailedImages": 0
        }

    def create_folders(self):
        if not os.path.exists('./images'):
            os.mkdir('./images')

        if not os.path.exists(f'./images/{self.collection_slug}'):
            os.mkdir(f'./images/{self.collection_slug}')

        if not os.path.exists(f'./images/{self.collection_slug}/image_data'):
            os.mkdir(f'./images/{self.collection_slug}/image_data')


    def get_total_supply(self):
        colletion_url = f"https://api.opensea.io/api/v1/collection/{self.collection_slug}/stats"

        response = requests.request("GET", colletion_url, headers=self.headers)
        if response.status_code == 200:
            supply = response.json()["stats"]["total_supply"]
            
            return int(supply)
        elif response.status_code == 404:
            print(f"Collection not found. Verify collection slug and try again.")
            raise ValueError()
        else:
            print(f"Unable to retrieve collection supply count. {response}")
            
    def retrieve_assets(self, cursor=None):
        asset_url = f"https://api.opensea.io/api/v1/assets?collection={self.collection_slug}&order_direction=desc&limit=50&include_orders=false"
        if (cursor):
            asset_url += f"&cursor={cursor}"

        response = requests.request("GET", asset_url, headers=self.headers)

        if (response.status_code == 200):
            response_json = response.json()

            for asset in response_json["assets"]:
                formatted_number = f"{int(asset['token_id']):04d}"
                print(f"\n#{formatted_number}:")
                # Check if data for the NFT already exists, if it does, skip saving it
                if os.path.exists(f'./images/{self.collection_slug}/image_data/{formatted_number}.json'):
                    print(f"  Data  -> [\u2713] (Already Downloaded)")
                    self.stats["AlreadyDownloadedData"] += 1
                else:
                    # Take the JSON from the URL, and dump it to the respective file.
                    dfile = open(f"./images/{self.collection_slug}/image_data/{formatted_number}.json", "w+")
                    json.dump(asset, dfile, indent=3)
                    dfile.close()
                    print(f"  Data  -> [\u2713] (Successfully downloaded)")
                    self.stats["DownloadedData"] += 1
                
                # Check if image already exists, if it does, skip saving it
                if os.path.exists(f'./images/{self.collection_slug}/{formatted_number}.png'):
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
                        file = open(f"./images/{self.collection_slug}/{formatted_number}.png", "wb+")
                        file.write(image.content)
                        file.close()
                        print(f"  Image -> [\u2713] (Successfully downloaded)")
                        self.stats["DownloadedImages"] += 1
                    # If the URL returns a status code other than "200 Successful", alert the user and don't save the image
                    else:
                        print(f"  Image -> [!] (HTTP Status {image.status_code})")
                        self.stats["FailedImages"] += 1
                        continue

            return response_json["next"]
        
        else:
            raise ValueError()

    def upload_assets(self):
        # create zip archive
        shutil.make_archive(f'{self.collection_slug}_archive', 'zip', f'./images/{self.collection_slug}')

    def print_stats(self, supply):
        print(f"""
        Finished downloading collection.

        Statistics
        -=-=-=-=-=-

        Total of {supply} units in collection "{self.collection_slug}".

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
            self.create_folders()
            supply = self.get_total_supply()
            print(f"Total Supply: {supply}")

            min_calls = math.ceil(supply / 50)
            cursor = None
            for _ in range(min_calls):
                cursor = self.retrieve_assets(cursor)
            self.upload_assets()
            self.print_stats(supply)
        except Exception as e:
            print(e)
            print(f"Encountered error. See above for details.")

if __name__ == "__main__":
    SeaportPirate(collection, api_key).main()
