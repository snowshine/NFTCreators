#!/bin/bash
S3_FOLDER_ROOT="s3://nf2-lambdas/"
PACKAGE_FOLDER="package"
VALIDATION_PACKAGE_ZIP="lambda-validation-package.zip"

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
    echo "Not implemented"
    exit 1
fi

if [ $1 == "cloudformation" ]
then
    echo "Deploying cloudformation template..."
    aws cloudformation deploy --region us-east-2 --profile nf2 --template-file data-arch.yaml --stack-name NF2Data --capabilities CAPABILITY_IAM
    exit 1
fi

echo "Provided deployment not found."
