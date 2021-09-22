CKPT=$1
mkdir -p checkpoints/downloads/$CKPT
rsync -avtP -e 'ssh -p 6001' root@10.176.52.112:/nfs/dubhe-prod/dataset/202/versionFile/V0001/yfshao/checkpoints/$CKPT checkpoints/downloads/$CKPT
