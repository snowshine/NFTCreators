import os
import time

def create_model_path_from_slug(collectionSlug):
    try:
        os.makedirs(f'./models/{collectionSlug}')
    except FileExistsError:
        pass

def create_output_image_path():
    try:
        os.makedirs(f'./tmp')
    except FileExistsError:
        pass

def get_model_path_from_slug(collectionSlug):
    return f'./models/{collectionSlug}/{collectionSlug}_generator.zip'

def get_model_directory_from_slug(collectionSlug):
    return f'./models/{collectionSlug}/{collectionSlug}_generator'

def get_model_to_load_from_slug(collectionSlug):
    return f'./models/{collectionSlug}/{collectionSlug}_generator/generator'

def model_does_exist(collectionSlug):
    if not os.path.exists(get_model_directory_from_slug(collectionSlug)):
        return False

    return True

def get_local_model_creation_time(collectionSlug):
    modelPath = get_model_directory_from_slug(collectionSlug)

    return os.path.getmtime(modelPath)
    