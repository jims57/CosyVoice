# docker login --username=jimmy@1904289922758417 crpi-e59jvk2aewsjh4xc.cn-shanghai.personal.cr.aliyuncs.com

docker tag jims57/cosy-tts:v1.3 crpi-e59jvk2aewsjh4xc.cn-shanghai.personal.cr.aliyuncs.com/watchfun-fc3-space/cosy-tts:v1.3
docker push crpi-e59jvk2aewsjh4xc.cn-shanghai.personal.cr.aliyuncs.com/watchfun-fc3-space/cosy-tts:v1.3

# docker pull crpi-e59jvk2aewsjh4xc.cn-shanghai.personal.cr.aliyuncs.com/watchfun-fc3-space/coqui-tts:v1.3

# version="1.2.1" && server_name="cosy-tts" && image_prefix="crpi-e59jvk2aewsjh4xc.cn-shanghai.personal.cr.aliyuncs.com/watchfun-fc3-space" && image_name="${image_prefix}/${server_name}" && tag="v${version}" && docker rm "${server_name}" -f && docker run --gpus all -it -p 9000:9000 -p 9002:9002 -p 9003:9003 --shm-size=24g -w ~/TTS --name "${server_name}" "${image_name}:${tag}" /bin/bash -c "bash /root/.jupyter_startup.sh; /bin/bash"

# version="1.2.1" && server_name="cosy-tts" && image_prefix="crpi-e59jvk2aewsjh4xc.cn-shanghai.personal.cr.aliyuncs.com/watchfun-fc3-space" && image_name="${image_prefix}/${server_name}" && tag="v${version}" && docker run --gpus all -it -p 9000:9000 -p 9002:9002 -p 9003:9003 --shm-size=24g -w ~/TTS --name "${server_name}" "${image_name}:${tag}" /bin/bash -c "bash /root/.jupyter_startup.sh; /bin/bash"