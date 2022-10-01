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
