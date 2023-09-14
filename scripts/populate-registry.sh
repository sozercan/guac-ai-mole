#!/usr/bin/env bash

set -euxo pipefail

# fail fast if REGISTRY env var is not set
if [ -z ${REGISTRY+x} ]; then
    echo "REGISTRY is not set"
    exit 1
fi

# array of image names
declare -a images=(
    "docker.io/library/nginx:1.19.10"
    "docker.io/library/ubuntu:22.04"
    "docker.io/library/debian:bookworm"
    "docker.io/library/alpine:3.14.2"
    "registry.k8s.io/coredns/coredns:v1.10.1"
    "registry.k8s.io/etcd:3.5.9-0"
    "registry.k8s.io/kube-apiserver:v1.28.0"
    "registry.k8s.io/kube-controller-manager:v1.28.0"
    "registry.k8s.io/kube-proxy:v1.28.0"
    "registry.k8s.io/kube-scheduler:v1.28.0"
)

# loop through the array
for image in "${images[@]}"
do
    slash_count=$(echo $image | grep -o '/' | wc -l)
    if [ $slash_count -eq 2 ]; then
        # image contains a repo ("docker.io/library/alpine:3.14.2")
        image_name=$(echo $image | cut -d'/' -f3 | cut -d':' -f1)
    else
        # image does not contain a repo ("registry.k8s.io/kube-apiserver:v1.28.0")
        image_name=$(echo $image | cut -d'/' -f2 | cut -d':' -f1)
    fi

    # get the image tag
    image_tag=$(echo $image | cut -d':' -f2)

    oras cp $image ${REGISTRY}/${image_name}:${image_tag}

    # generate sboms
    # TODO: only generates for running platform today, not all platforms
    syft $image --output spdx-json --file _output/${image_name}-${image_tag}.json

    # attach the sbom to the image
    oras attach --artifact-type application/spdx+json ${REGISTRY}/${image_name}:${image_tag} _output/${image_name}-${image_tag}.json:application/json
done
