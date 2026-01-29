#IMAGE_NAME=nvcr.io/nvidia/pyg:24.07-py3
IMAGE_NAME="registry.rcp.epfl.ch/imos-ksteiner/base_image:latest"
export $(cat .rcp.env | xargs)
#--cpu 6 \
#--memory 32 \
export RCP_CAAS_LABSCRATCH=imos-scratch
COMMAND="cd /scratch/imos-students/ksteiner/BeamElastoDynamics && python run_passive_train.py --volume-loss-weight=0.0 --epochs=500"

runai submit \
  --name train-volume-tru-0-500 \
  --image ${IMAGE_NAME} \
  --gpu 1 \
  --run-as-uid ${LDAP_UID} \
  --run-as-gid ${LDAP_GID} \
  --existing-pvc "claimname=${RCP_CAAS_LABSCRATCH},path=/scratch" \
  --large-shm \
  --command -- /bin/bash -c "$COMMAND"
