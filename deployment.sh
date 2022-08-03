# https://towardsdatascience.com/to-serve-man-60246a82d953

docker build -t my-model:0.2 .
docker tag my-model:0.2 jsilverstein/my-model:0.2
docker push jsilverstein/my-model:0.2
docker run -d --rm --name my-model -p 9000:9000 my-model:0.2
# docker logs -f my-model

# test docker container server
# python3 my-model-client.py http://localhost:9000/api/v0.1/predictions

kubectl create -f seldon-deploy.yaml

# test Kubernetes service
kubectl port-forward svc/my-model-svc 8000:8000
python3 my-model-client.py http://localhost:8000/api/v0.1/predictions 