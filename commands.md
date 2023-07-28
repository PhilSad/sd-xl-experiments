# sd-xl-experiments

## gcloud setup

echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-cli -y

gcloud login

gsutil cp gs://anyoneapps/models/sd_xl_base_0.9.safetensors .

## setup vscode on runpod

apt-get update
apt-get install wget gpg
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg
sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
rm -f packages.microsoft.gpg

apt install apt-transport-https
apt update
apt install code


adduser vscode
usermod -aG sudo vscode

su vscode


## config files
### dataset
```toml
[[datasets]]
resolution = [1024, 1024]
batch_size = 4
keep_tokens = 2

  [[datasets.subsets]]
  image_dir = '/workspace/data/man'
  class_tokens = 'shs man'

[general]
keep_tokens = 1
resolution = [1024, 1024]
```

### config
```toml
[model_arguments]
v2 = false
v_parameterization = false
pretrained_model_name_or_path = "./sd_xl_base_0.9.safetensors"

[optimizer_arguments]
optimizer_type = "adafactor"
optimizer_args = [ "scale_parameter=False", "relative_step=False", "warmup_init=False" ]
lr_scheduler = "constant_with_warmup"
lr_warmup_steps = 100
learning_rate = 4e-7 # SDXL original learning rate
max_grad_norm = 0.0
train_text_encoder = false

[dataset_arguments]
debug_dataset = false

[training_arguments]
output_dir = "output"
output_name = "last"
save_precision = "fp16"
save_n_epoch_ratio = 1
save_state = false
train_batch_size = 1
mem_eff_attn = false
max_train_steps = 1000
max_data_loader_n_workers = 1
persistent_data_loader_workers = true
gradient_checkpointing = true
gradient_accumulation_steps = 2
mixed_precision = "bf16"
logging_dir = "logs"
log_prefix = "last"

[sample_prompt_arguments]
sample_every_n_steps = 661
sample_sampler = "ddim"

[saving_arguments]
save_model_as = "safetensors"
```

### Train!
```
!accelerate launch --num_cpu_threads_per_process 1 /workspace/sd-scripts/sdxl_train.py \
    --config_file=/workspace/config.toml\
    --dataset_config=/workspace/config_dataset.toml \
    --xformers \
    --cache_latents \
    --full_bf16\
    --cache_text_encoder_outputs
```
