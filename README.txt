This app uses ML to predict whether a movie review is possitive or negative.

Before doing anything you must have docker installed and some way to run makefiles such as WSL. Instructions for installing WSL can be found here: https://learn.microsoft.com/en-us/windows/wsl/install
I would also recommend installing postman. Postman can be downloaded here: https://www.postman.com/downloads/

Open a linux terminal wherever the files are saved.

Then run the following commands in order:
make build
make run

from the same window  you can run python evaluate.py to test the code.

after this from another terminal or in postman you can make send commands. The commands you can send are:
post 0.0.0.0:8000/predict

for the two predict commands your body should look like this: {"text": "My movie review text here."} if you want to know what a command should look like veiw test.json

run: make clean to remove the docker container and image.
it may leave files in wsl this can be fixed by rebooting wsl and docker