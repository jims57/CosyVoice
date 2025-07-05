# save container to image
version="1.3.1" && server_name="cosy-tts" && registry_url="crpi-e59jvk2aewsjh4xc.cn-shanghai.personal.cr.aliyuncs.com/watchfun-fc3-space" && image_name="${registry_url}/${server_name}" && tag="v${version}" && docker commit --pause=false "${server_name}" "${image_name}:${tag}"

# push image to acr
docker push "${registry_url}/cosy-tts:v${version}"