This repo represents the backend (and probably the frontend) service for generating face landmarks from audio.

# Docker Setup
1. (Optional) Train a model using the [training code](https://github.com/audiovtuber/Talking-Face-Landmarks-from-Speech) and export it as a TorchScript model. Copy the trained torchscript model into this repo's project root with the name `torchscript_model.pt`
> A sample model is included in this repository already. It's pretty terrible, but useful for development
2. Clone this repo
3. `cd` to the project root, then `chmod +x run.sh`
4. Run `./run.sh`
> Step 4 will build the service's docker container and run it locally for you

## Developing
Code changes are **not currently hot-loaded** into the container. Therefore, re-run `./run.sh` if you make any changes to code, the dockerfile, or if you replace the TorchScript model

## Deploying
First, complete the ["one time setup"](#one-time-setup) below if you haven't already. After that, simply run [`deploy.sh`](deploy.sh) to deploy the service to Google Cloud Run! Note that each time you run the deploy script, it will generate a new revision of the service and push a new container to Google Artifact Registry, which may incur costs
### One-Time Setup
If you don't have `gcloud` installed, simply run [`configure_gcp.sh`](configure_gcp.sh). It will install gcloud and then configure it for this project. Since you likely don't have access to the specified Google Cloud project, you'll likely need to change the project name and configure docker to use your own artifact registry
