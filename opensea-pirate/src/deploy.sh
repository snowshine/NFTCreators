#!/bin/bash
DEPLOYMENT_REGION="us-east-2"
S3_FOLDER_ROOT="s3://nf2-lambdas/"
PACKAGE_FOLDER="package"
VALIDATION_PACKAGE_ZIP="lambda-validation-package.zip"
DOCKER_VERSION="v0.0.1"

if [ -z "$1" ]
  then
    echo "No argument supplied. Specify lambda to deploy from list:"
    echo " - cloudformation"
    echo " - docker"
    echo " - validate-lambda"
    exit 0
fi

cleanup_artifacts() {
    if [ -d "$PACKAGE_FOLDER" ]; then
        echo "Removing existing package folder..."
        rm -rf "$PACKAGE_FOLDER"
    fi
    if [ -f "$VALIDATION_PACKAGE_ZIP" ]; then
        echo "Removing existing package zip..."
        rm -f "$VALIDATION_PACKAGE_ZIP"
    fi
}

# cleanup any artifacts that may be left over
cleanup_artifacts

if [ $1 == "validate-lambda" ]
then
    # install all python requirements
    pip3 install --target ./package -r requirements.txt > /dev/null

    # create package zip
    cd ./package
    zip -r "../$VALIDATION_PACKAGE_ZIP" . > /dev/null

    # add lambda to the package
    cd ..
    zip -g "$VALIDATION_PACKAGE_ZIP" validate-slug.py > /dev/null

    # add utils to the package
    zip -ur "$VALIDATION_PACKAGE_ZIP" ./utils > /dev/null

    # upload package to s3
    echo "Uploading zip to S3..."
    aws s3 cp "./$VALIDATION_PACKAGE_ZIP" $S3_FOLDER_ROOT --profile nf2 > /dev/null
    cleanup_artifacts

    echo "Package location:"
    echo "    $S3_FOLDER_ROOT$VALIDATION_PACKAGE_ZIP"
    exit 1
fi

if [ $1 == "docker" ]
then
    if [ -z $2 ]
    then
        echo "No account id provide. Unable to deploy Dockerfile."
        exit 0
    fi

    docker build --tag $DOCKER_VERSION .

    aws ecr get-login-password --profile nf2 --region $DEPLOYMENT_REGION | docker login --username AWS --password-stdin "$2.dkr.ecr.$DEPLOYMENT_REGION.amazonaws.com"
    tagged=$(docker images | grep $DOCKER_VERSION | head -n 1 | tr -s ' ' | cut -d ' ' -f 3)
    
    docker tag $tagged "$2.dkr.ecr.$DEPLOYMENT_REGION.amazonaws.com/nf2-dev:$DOCKER_VERSION"

    docker push "$2.dkr.ecr.$DEPLOYMENT_REGION.amazonaws.com/nf2-dev:$DOCKER_VERSION"

    exit 1
fi

if [ $1 == "cloudformation" ]
then
    echo "Deploying cloudformation template..."
    aws cloudformation deploy --region $DEPLOYMENT_REGION --profile nf2 --template-file data-arch.yaml --stack-name NF2Data --capabilities CAPABILITY_IAM
    exit 1
fi

echo "Provided deployment not found."
