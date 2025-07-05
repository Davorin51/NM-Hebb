# iz korijena projekta (gdje su Dockerfile i VGG_SGD_Hebb.py)
docker build -t hebbian-mobilenet-cifar100 .

# pokreni s pristupom svim GPU-ovima i mapiranim rezultatskim direktorijem
docker run --gpus all \
           -v $(pwd)/results_exp_vgg:/workspace/results_exp_vgg \
           hebbian-mobilenet-cifar100


# Ako ovo ne bude radilo u dokeru, treba samo gore ove reqs instalirat (mogu i najnoviji raditi će.)