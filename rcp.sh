#IMAGE_NAME=nvcr.io/nvidia/pyg:24.07-py3
IMAGE_NAME="registry.rcp.epfl.ch/imos-ksteiner/base_image:latest"
export $(cat .rcp.env | xargs)
#--cpu 6 \
#--memory 32 \
runai delete job interactive-test
runai submit \
    --name interactive-test \
    -i $IMAGE_NAME \
    --gpu 1 \
    --cpu 16 \
    --memory 64 \
    --interactive \
    --run-as-uid ${LDAP_UID} \
    --run-as-gid ${LDAP_GID} \
    --existing-pvc "claimname=imos-scratch,path=/scratch" \
    --attach