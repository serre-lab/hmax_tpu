# Create kubernetes cluster with 4 nodes for self-sup experiments
CLUSTER_NAME=herbarium
MIN_NODES=0
MAX_NODES=1  # 16
START_NODES=1  # 8

gcloud container --project "beyond-dl-1503610372419" \
clusters create "$CLUSTER_NAME" \
--zone "europe-west4-a" --no-enable-basic-auth \
--cluster-version "1.20.8-gke.900" \
--machine-type "n1-standard-4" \
--image-type "UBUNTU" \
--disk-type "pd-standard" \
--disk-size "20" \
--scopes "https://www.googleapis.com/auth/cloud-platform" --preemptible \
--max-nodes "$MAX_NODES" \
--min-nodes "$MIN_NODES" \
--num-nodes "$START_NODES" \
--network "projects/beyond-dl-1503610372419/global/networks/default" \
--subnetwork "projects/beyond-dl-1503610372419/regions/europe-west4/subnetworks/default" \
--enable-ip-alias \
--enable-tpu


# Apply a quota to the cluster that matches preemptible resources
# kubectl create namespace quota-pod
# kubectl apply -f quota.yaml --namespace=quota-pod
# gcloud beta container clusters update $CLUSTER_NAME --autoscaling-profile optimize-utilization