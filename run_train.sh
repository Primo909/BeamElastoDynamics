#IMAGE_NAME=nvcr.io/nvidia/pyg:24.07-py3
IMAGE_NAME="registry.rcp.epfl.ch/imos-ksteiner/base_image:latest"
export $(cat .rcp.env | xargs)
#--cpu 6 \
#--memory 32 \
export RCP_CAAS_LABSCRATCH=imos-scratch
COMMAND="cd /scratch/imos-students/ksteiner/BeamElastoDynamics && python run_passive_train.py --volume-loss-weight=0.3 --epochs=700 --resume-from=saved_models/2026-01-29_13-17-26/Epoch_160_GenLoss_0.0130941458.pth"

runai submit \
  --name train-volume-resume-160 \
  --image ${IMAGE_NAME} \
  --gpu 1 \
  --run-as-uid ${LDAP_UID} \
  --run-as-gid ${LDAP_GID} \
  --existing-pvc "claimname=${RCP_CAAS_LABSCRATCH},path=/scratch" \
  --large-shm \
  --command -- /bin/bash -c "$COMMAND"
