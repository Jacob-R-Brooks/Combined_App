Before doing anything you must have docker installed and some way to run makefiles such as WSL. Instructions for installing WSL can be found here: https://learn.microsoft.com/en-us/windows/wsl/install
I would also recommend installing postman. Postman can be downloaded here: https://www.postman.com/downloads/

Open a linux terminal wherever the files are saved.

Then run the following commands in order:
make build
make run

after this from another terminal or in postman you can make send commands. The commands you can send are:

/health
/predict (requires a body)
/predict_proba (requires a body)
/example

your commands should look like this in postman: get 0.0.0.0:8000/health or post 0.0.0.0:8000/predict

for the two predict commands your body should look like this: {"text": "My movie review text here."}

when you are done using the fastAPI app you can use control+c to end the app

run: make clean to remove the docker container and image.