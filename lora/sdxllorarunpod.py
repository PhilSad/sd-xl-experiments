from IPython.display import clear_output
from subprocess import call, getoutput, Popen
from IPython.display import display
import ipywidgets as widgets
import io
from PIL import Image, ImageDraw, ImageOps
import fileinput
import time
import os
from os import listdir
from os.path import isfile
import random
import sys
from io import BytesIO
import requests
from collections import defaultdict
from math import log, sqrt
import numpy as np
import sys
import fileinput
from subprocess import check_output
import six
import base64

from urllib.parse import urlparse, parse_qs, unquote
import urllib.request
from urllib.request import urlopen, Request

import tempfile
from tqdm import tqdm




def Deps(force_reinstall):

    if not force_reinstall and os.path.exists('/usr/local/lib/python3.10/dist-packages/safetensors'):
        ntbks()
        call('pip install --root-user-action=ignore --disable-pip-version-check -qq diffusers==0.18.1', shell=True, stdout=open('/dev/null', 'w'))
        print('[1;32mModules and notebooks updated, dependencies already installed')        
        os.environ['TORCH_HOME'] = '/workspace/cache/torch'
        os.environ['PYTHONWARNINGS'] = 'ignore'
    else:
        call('pip install --root-user-action=ignore --disable-pip-version-check --no-deps -qq gdown PyWavelets numpy==1.23.5 accelerate==0.12.0 --force-reinstall', shell=True, stdout=open('/dev/null', 'w'))
        ntbks()
        if os.path.exists('deps'):
            call("rm -r deps", shell=True)
        if os.path.exists('diffusers'):
            call("rm -r diffusers", shell=True)
        call('mkdir deps', shell=True)
        if not os.path.exists('cache'):
            call('mkdir cache', shell=True)
        os.chdir('deps')
        dwn("https://huggingface.co/TheLastBen/dependencies/resolve/main/rnpddeps-t2.tar.zst", "/workspace/deps/rnpddeps-t2.tar.zst", "Installing dependencies")
        call('tar -C / --zstd -xf rnpddeps-t2.tar.zst', shell=True, stdout=open('/dev/null', 'w'))
        call("sed -i 's@~/.cache@/workspace/cache@' /usr/local/lib/python3.10/dist-packages/transformers/utils/hub.py", shell=True)
        os.chdir('/workspace')
        call('pip install --root-user-action=ignore --disable-pip-version-check -qq diffusers==0.18.1', shell=True, stdout=open('/dev/null', 'w'))
        call("git clone --depth 1 -q --branch main https://github.com/TheLastBen/diffusers", shell=True, stdout=open('/dev/null', 'w'))
        #call('pip install --root-user-action=ignore --disable-pip-version-check -qq gradio==3.39.0', shell=True, stdout=open('/dev/null', 'w'))
        call("rm -r deps", shell=True)
        os.chdir('/workspace')
        os.environ['TORCH_HOME'] = '/workspace/cache/torch'
        os.environ['PYTHONWARNINGS'] = 'ignore'
        call("sed -i 's@text = _formatwarnmsg(msg)@text =\"\"@g' /usr/lib/python3.10/warnings.py", shell=True)
        clear_output()

        done()


def dwn(url, dst, msg):
    file_size = None
    req = Request(url, headers={"User-Agent": "torch.hub"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    with tqdm(total=file_size, disable=False, mininterval=0.5,
              bar_format=msg+' |{bar:20}| {percentage:3.0f}%') as pbar:
        with open(dst, "wb") as f:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                pbar.update(len(buffer))
            f.close()


def ntbks():

    os.chdir('/workspace')
    if not os.path.exists('Latest_Notebooks'):
        call('mkdir Latest_Notebooks', shell=True)
    else:
        call('rm -r Latest_Notebooks', shell=True)
        call('mkdir Latest_Notebooks', shell=True)
    os.chdir('/workspace/Latest_Notebooks')
    call('wget -q -i https://huggingface.co/datasets/TheLastBen/RNPD/raw/main/Notebooks.txt', shell=True)
    call('rm Notebooks.txt', shell=True)
    os.chdir('/workspace')

def done():
    done = widgets.Button(
        description='Done!',
        disabled=True,
        button_style='success',
        tooltip='',
        icon='check'
    )
    display(done)



def mdlvxl():

  os.chdir('/workspace')

  if os.path.exists('stable-diffusion-XL') and not os.path.exists('/workspace/stable-diffusion-XL/unet/diffusion_pytorch_model.safetensors'):
     call('rm -r stable-diffusion-XL', shell=True)
  if not os.path.exists('stable-diffusion-XL'):
      print('[1;33mDownloading SDXL model...')
      call('mkdir stable-diffusion-XL', shell=True)
      os.chdir('stable-diffusion-XL')
      call('git init', shell=True, stdout=open('/dev/null', 'w'), stderr=open('/dev/null', 'w'))
      call('git lfs install --system --skip-repo', shell=True, stdout=open('/dev/null', 'w'), stderr=open('/dev/null', 'w'))
      call('git remote add -f origin  https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0', shell=True, stdout=open('/dev/null', 'w'), stderr=open('/dev/null', 'w'))
      call('git config core.sparsecheckout true', shell=True, stdout=open('/dev/null', 'w'), stderr=open('/dev/null', 'w'))
      call('echo -e "\nscheduler\ntext_encoder\ntext_encoder_2\ntokenizer\ntokenizer_2\nunet\nvae\nfeature_extractor\nmodel_index.json\n!*.safetensors\n!*.bin\n!*.onnx*\n!*.xml" > .git/info/sparse-checkout', shell=True, stdout=open('/dev/null', 'w'), stderr=open('/dev/null', 'w'))
      call('git pull origin main', shell=True, stdout=open('/dev/null', 'w'), stderr=open('/dev/null', 'w'))
      dwn('https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder/model.safetensors', 'text_encoder/model.safetensors', '1/4')
      dwn('https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder_2/model.safetensors', 'text_encoder_2/model.safetensors', '2/4')
      dwn('https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/vae/diffusion_pytorch_model.safetensors', 'vae/diffusion_pytorch_model.safetensors', '3/4')
      dwn('https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/unet/diffusion_pytorch_model.safetensors', 'unet/diffusion_pytorch_model.safetensors', '4/4')  
      call('rm -r .git', shell=True, stdout=open('/dev/null', 'w'), stderr=open('/dev/null', 'w'))
      os.chdir('/workspace')
      clear_output()
      while not os.path.exists('/workspace/stable-diffusion-XL/unet/diffusion_pytorch_model.safetensors'):
            print('[1;31mInvalid HF token, make sure you have access to the model')
            time.sleep(8)
      if os.path.exists('/workspace/stable-diffusion-XL/unet/diffusion_pytorch_model.safetensors'):
          print('[1;32mUsing SDXL model')
  else:
    print('[1;32mUsing SDXL model')
    
  call("sed -i 's@\"force_upcast.*@@' /workspace/stable-diffusion-XL/vae/config.json", shell=True)

  

def downloadmodel_hfxl(Path_to_HuggingFace):

  os.chdir('/workspace')
  if os.path.exists('stable-diffusion-custom'):
    call("rm -r stable-diffusion-custom", shell=True)
  clear_output()
  
  if os.path.exists('Fast-Dreambooth/token.txt'):
    with open("Fast-Dreambooth/token.txt") as f:
       token = f.read()
    authe=f'https://USER:{token}@'
  else:
    authe="https://"

  clear_output()
  call("mkdir stable-diffusion-custom", shell=True)
  os.chdir("stable-diffusion-custom")
  call("git init", shell=True)
  call("git lfs install --system --skip-repo", shell=True)
  call('git remote add -f origin '+authe+'huggingface.co/'+Path_to_HuggingFace, shell=True)
  call("git config core.sparsecheckout true", shell=True)
  call('echo -e "\nscheduler\ntext_encoder\ntokenizer\nunet\nvae\nfeature_extractor\nmodel_index.json\n!*.fp16.safetensors" > .git/info/sparse-checkout', shell=True)
  call("git pull origin main", shell=True)
  if os.path.exists('unet/diffusion_pytorch_model.safetensors'):
    call("rm -r .git", shell=True)
    os.chdir('/workspace')
    clear_output()
    done()
  while not os.path.exists('/workspace/stable-diffusion-custom/unet/diffusion_pytorch_model.safetensors'):
        print('[1;31mCheck the link you provided')
        os.chdir('/workspace')
        time.sleep(5)
  
        

def downloadmodel_link_xl(MODEL_LINK): 

    import wget
    import gdown
    from gdown.download import get_url_from_gdrive_confirmation    
    
    def getsrc(url):
        parsed_url = urlparse(url)
        if parsed_url.netloc == 'civitai.com':
            src='civitai'
        elif parsed_url.netloc == 'drive.google.com':
            src='gdrive'
        elif parsed_url.netloc == 'huggingface.co':
            src='huggingface'
        else:
            src='others'
        return src
        
    src=getsrc(MODEL_LINK)

    def get_name(url, gdrive):
        if not gdrive:
            response = requests.get(url, allow_redirects=False)
            if "Location" in response.headers:
                redirected_url = response.headers["Location"]
                quer = parse_qs(urlparse(redirected_url).query)
                if "response-content-disposition" in quer:
                    disp_val = quer["response-content-disposition"][0].split(";")
                    for vals in disp_val:
                        if vals.strip().startswith("filename="):
                            filenm=unquote(vals.split("=", 1)[1].strip())
                            return filenm.replace("\"","")
        else:
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"}
            lnk="https://drive.google.com/uc?id={id}&export=download".format(id=url[url.find("/d/")+3:url.find("/view")])
            res = requests.session().get(lnk, headers=headers, stream=True, verify=True)
            res = requests.session().get(get_url_from_gdrive_confirmation(res.text), headers=headers, stream=True, verify=True)
            content_disposition = six.moves.urllib_parse.unquote(res.headers["Content-Disposition"])
            filenm = re.search(r"filename\*=UTF-8''(.*)", content_disposition).groups()[0].replace(os.path.sep, "_")
            return filenm   

    if src=='civitai':
       modelname=get_name(MODEL_LINK, False)
    elif src=='gdrive':
       modelname=get_name(MODEL_LINK, True)
    else:
       modelname=os.path.basename(MODEL_LINK)


    os.chdir('/workspace')
    if src=='huggingface':
        dwn(MODEL_LINK, modelname,'[1;33mDownloading the Model')
    else:
        call("gdown --fuzzy " +MODEL_LINK+ " -O "+modelname, shell=True)
    
    if os.path.exists(modelname):
      if os.path.getsize(modelname) > 1810671599:
          
        print('[1;32mConverting to diffusers...')    
        call('python /workspace/diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path '+modelname+' --dump_path stable-diffusion-custom --from_safetensors', shell=True, stdout=open('/dev/null', 'w'), stderr=open('/dev/null', 'w'))
       
        if os.path.exists('stable-diffusion-custom/unet/diffusion_pytorch_model.bin'):
          os.chdir('/workspace')
          clear_output()
          done()
        else:
            while not os.path.exists('stable-diffusion-custom/unet/diffusion_pytorch_model.bin'):
              print('[1;31mConversion error')
              os.chdir('/workspace')
              time.sleep(5)
    else:
        while os.path.getsize(modelname) < 1810671599:
           print('[1;31mWrong link, check that the link is valid')
           os.chdir('/workspace')
           time.sleep(5)



def downloadmodel_path_xl(MODEL_PATH):

  import wget
  os.chdir('/workspace')
  clear_output() 
  if os.path.exists(str(MODEL_PATH)):
  
    print('[1;32mConverting to diffusers...')
    call('python /workspace/diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path '+MODEL_PATH+' --dump_path stable-diffusion-custom --from_safetensors', shell=True, stdout=open('/dev/null', 'w'), stderr=open('/dev/null', 'w'))

    if os.path.exists('stable-diffusion-custom/unet/diffusion_pytorch_model.bin'):
      clear_output()
      done()
    while not os.path.exists('stable-diffusion-custom/unet/diffusion_pytorch_model.bin'):
      print('[1;31mConversion error')
      os.chdir('/workspace')
      time.sleep(5)
  else:
    while not os.path.exists(str(MODEL_PATH)):
       print('[1;31mWrong path, use the file explorer to copy the path')
       os.chdir('/workspace')
       time.sleep(5)




def dls_xlf(Path_to_HuggingFace, MODEL_PATH, MODEL_LINK):

    os.chdir('/workspace')
    
    if Path_to_HuggingFace != "":
      downloadmodel_hfxl(Path_to_HuggingFace)
      MODEL_NAMExl="/workspace/stable-diffusion-custom"

    elif MODEL_PATH !="":
        
      downloadmodel_path_xl(MODEL_PATH)
      MODEL_NAMExl="/workspace/stable-diffusion-custom"

    elif MODEL_LINK !="":

      downloadmodel_link_xl(MODEL_LINK)
      MODEL_NAMExl="/workspace/stable-diffusion-custom"

    else:
        mdlvxl()
        MODEL_NAMExl="/workspace/stable-diffusion-XL"

    return MODEL_NAMExl    
    


def sess_xl(Session_Name, MODEL_NAMExl):
    import gdown
    import wget
    os.chdir('/workspace')
    PT=""

    while Session_Name=="":
      print('[1;31mInput the Session Name:') 
      Session_Name=input("")
    Session_Name=Session_Name.replace(" ","_")

    WORKSPACE='/workspace/Fast-Dreambooth'

    INSTANCE_NAME=Session_Name
    OUTPUT_DIR="/workspace/models/"+Session_Name
    SESSION_DIR=WORKSPACE+"/Sessions/"+Session_Name
    INSTANCE_DIR=SESSION_DIR+"/instance_images"
    CAPTIONS_DIR=SESSION_DIR+'/captions'
    MDLPTH=str(SESSION_DIR+"/"+Session_Name+'.safetensors')
    

    if os.path.exists(str(SESSION_DIR)) and not os.path.exists(MDLPTH):
        print('[1;32mLoading session with no previous LoRa model')
        if MODEL_NAMExl=="":
            print('[1;31mNo model found, use the "Model Download" cell to download a model.')
        else:
            print('[1;32mSession Loaded, proceed')

    elif not os.path.exists(str(SESSION_DIR)):
        call('mkdir -p '+INSTANCE_DIR, shell=True)
        print('[1;32mCreating session...')
        if MODEL_NAMExl=="":
          print('[1;31mNo model found, use the "Model Download" cell to download a model.')
        else:
          print('[1;32mSession created, proceed to uploading instance images')
          if MODEL_NAMExl=="":
             print('[1;31mNo model found, use the "Model Download" cell to download a model.')        

    else:
        print('[1;32mSession Loaded, proceed')
        
        
    return WORKSPACE, Session_Name, INSTANCE_NAME, OUTPUT_DIR, SESSION_DIR, INSTANCE_DIR, CAPTIONS_DIR, MDLPTH, MODEL_NAMExl



def uplder(Remove_existing_instance_images, Crop_images, Crop_size, IMAGES_FOLDER_OPTIONAL, INSTANCE_DIR, CAPTIONS_DIR):

    if os.path.exists(INSTANCE_DIR+"/.ipynb_checkpoints"):
      call('rm -r '+INSTANCE_DIR+'/.ipynb_checkpoints', shell=True)

    uploader = widgets.FileUpload(description="Choose images",accept='image/*, .txt', multiple=True)
    Upload = widgets.Button(
        description='Upload',
        disabled=False,
        button_style='info', 
        tooltip='Click to upload the chosen instance images',
        icon=''
    )


    def up(Upload):
        with out: 
            uploader.close()
            Upload.close()
            upld(Remove_existing_instance_images, Crop_images, Crop_size, IMAGES_FOLDER_OPTIONAL, INSTANCE_DIR, CAPTIONS_DIR, uploader)
            done()
    out=widgets.Output()
    
    if IMAGES_FOLDER_OPTIONAL=="":
      Upload.on_click(up)
      display(uploader, Upload, out)
    else:
       upld(Remove_existing_instance_images, Crop_images, Crop_size, IMAGES_FOLDER_OPTIONAL, INSTANCE_DIR, CAPTIONS_DIR, uploader)
       done()
    
    

def upld(Remove_existing_instance_images, Crop_images, Crop_size, IMAGES_FOLDER_OPTIONAL, INSTANCE_DIR, CAPTIONS_DIR, uploader):

    from tqdm import tqdm
    if Remove_existing_instance_images:
        if os.path.exists(str(INSTANCE_DIR)):
            call("rm -r " +INSTANCE_DIR, shell=True)
        if os.path.exists(str(CAPTIONS_DIR)):
            call("rm -r " +CAPTIONS_DIR, shell=True)            


    if not os.path.exists(str(INSTANCE_DIR)):
        call("mkdir -p " +INSTANCE_DIR, shell=True)
    if not os.path.exists(str(CAPTIONS_DIR)):
        call("mkdir -p " +CAPTIONS_DIR, shell=True)        


    if IMAGES_FOLDER_OPTIONAL !="":
        if os.path.exists(IMAGES_FOLDER_OPTIONAL+"/.ipynb_checkpoints"):
          call('rm -r '+IMAGES_FOLDER_OPTIONAL+'/.ipynb_checkpoints', shell=True)    
    
        if any(file.endswith('.{}'.format('txt')) for file in os.listdir(IMAGES_FOLDER_OPTIONAL)):
            call('mv '+IMAGES_FOLDER_OPTIONAL+'/*.txt '+CAPTIONS_DIR, shell=True)
        if Crop_images:   
            os.chdir(str(IMAGES_FOLDER_OPTIONAL))
            call('find . -name "* *" -type f | rename ' "'s/ /-/g'", shell=True)
            os.chdir('/workspace')    
            for filename in tqdm(os.listdir(IMAGES_FOLDER_OPTIONAL), bar_format='  |{bar:15}| {n_fmt}/{total_fmt} Uploaded'):
                extension = filename.split(".")[-1]
                identifier=filename.split(".")[0]
                new_path_with_file = os.path.join(INSTANCE_DIR, filename)
                file = Image.open(IMAGES_FOLDER_OPTIONAL+"/"+filename)
                file=file.convert("RGB")
                file=ImageOps.exif_transpose(file)
                width, height = file.size           
                if file.size !=(Crop_size, Crop_size):
                    image=crop_image(file, Crop_size)
                    if extension.upper()=="JPG" or extension.upper()=="jpg":
                        image[0].save(new_path_with_file, format="JPEG", quality = 100)
                    else:
                        image[0].save(new_path_with_file, format=extension.upper())
                        
                else:
                   call("cp \'"+IMAGES_FOLDER_OPTIONAL+"/"+filename+"\' "+INSTANCE_DIR, shell=True)                        

        else:
            for filename in tqdm(os.listdir(IMAGES_FOLDER_OPTIONAL), bar_format='  |{bar:15}| {n_fmt}/{total_fmt} Uploaded'):
                call("cp -r " +IMAGES_FOLDER_OPTIONAL+"/. " +INSTANCE_DIR, shell=True)

    elif IMAGES_FOLDER_OPTIONAL =="":
        up=""  
        for file in uploader.value:
          filename = file['name']
          if filename.split(".")[-1]=="txt":
            with open(CAPTIONS_DIR+'/'+filename, 'w') as f:
                f.write(bytes(file['content']).decode())            
          up=[file for file in uploader.value if not file['name'].endswith('.txt')]
        if Crop_images:
            for file in tqdm(up, bar_format='  |{bar:15}| {n_fmt}/{total_fmt} Uploaded'):
                filename = file['name']
                img = Image.open(io.BytesIO(file['content']))
                img=img.convert("RGB")
                img=ImageOps.exif_transpose(img)
                extension = filename.split(".")[-1]
                identifier=filename.split(".")[0]

                if extension.upper()=="JPG" or extension.upper()=="jpg":
                    img.save(INSTANCE_DIR+"/"+filename, format="JPEG", quality = 100) 
                else:
                    img.save(INSTANCE_DIR+"/"+filename, format=extension.upper())                
                
                new_path_with_file = os.path.join(INSTANCE_DIR, filename)
                file = Image.open(new_path_with_file)
                width, height = file.size    
                if file.size !=(Crop_size, Crop_size):
                    image=crop_image(file, Crop_size)
                    if extension.upper()=="JPG" or extension.upper()=="jpg":
                        image[0].save(new_path_with_file, format="JPEG", quality = 100) 
                    else:
                        image[0].save(new_path_with_file, format=extension.upper())

        else:
            for file in tqdm(uploader.value, bar_format='  |{bar:15}| {n_fmt}/{total_fmt} Uploaded'):
                filename = file['name']
                img = Image.open(io.BytesIO(file['content']))
                img=img.convert("RGB")
                extension = filename.split(".")[-1]
                identifier=filename.split(".")[0]   
                
                if extension.upper()=="JPG" or extension.upper()=="jpg":
                    img.save(INSTANCE_DIR+"/"+filename, format="JPEG", quality = 100) 
                else:
                    img.save(INSTANCE_DIR+"/"+filename, format=extension.upper())                  
   

    os.chdir(INSTANCE_DIR)
    call('find . -name "* *" -type f | rename ' "'s/ /-/g'", shell=True)
    os.chdir(CAPTIONS_DIR)
    call('find . -name "* *" -type f | rename ' "'s/ /-/g'", shell=True)    
    os.chdir('/workspace')




def caption(CAPTIONS_DIR, INSTANCE_DIR):
   
  paths=""
  out=""
  widgets_l=""
  clear_output()
  def Caption(path):
      if path!="Select an instance image to caption":
        
        name = os.path.splitext(os.path.basename(path))[0]
        ext=os.path.splitext(os.path.basename(path))[-1][1:]
        if ext=="jpg" or "JPG":
          ext="JPEG"        

        if os.path.exists(CAPTIONS_DIR+"/"+name + '.txt'):
          with open(CAPTIONS_DIR+"/"+name + '.txt', 'r') as f:
              text = f.read()
        else:
          with open(CAPTIONS_DIR+"/"+name + '.txt', 'w') as f:
              f.write("")
              with open(CAPTIONS_DIR+"/"+name + '.txt', 'r') as f:
                  text = f.read()   

        img=Image.open(os.path.join(INSTANCE_DIR,path))
        img=img.convert("RGB")
        img=img.resize((420, 420))
        image_bytes = BytesIO()
        img.save(image_bytes, format=ext, qualiy=10)
        image_bytes.seek(0)
        image_data = image_bytes.read()
        img= image_data  
        image = widgets.Image(
            value=img,
            width=420,
            height=420
        )
        text_area = widgets.Textarea(value=text, description='', disabled=False, layout={'width': '300px', 'height': '120px'})
        

        def update_text(text):
            with open(CAPTIONS_DIR+"/"+name + '.txt', 'w') as f:
                f.write(text)

        button = widgets.Button(description='Save', button_style='success')
        button.on_click(lambda b: update_text(text_area.value))

        return widgets.VBox([widgets.HBox([image, text_area, button])])


  paths = os.listdir(INSTANCE_DIR)
  widgets_l = widgets.Select(options=["Select an instance image to caption"]+paths, rows=25)


  out = widgets.Output()

  def click(change):
      with out:
          out.clear_output()
          display(Caption(change.new))

  widgets_l.observe(click, names='value')
  display(widgets.HBox([widgets_l, out]))



def dbtrainxl(Unet_Training_Epochs, Text_Encoder_Training_Epochs, Unet_Learning_Rate, Text_Encoder_Learning_Rate, dim, Offset_Noise, Resolution, MODEL_NAME, SESSION_DIR, INSTANCE_DIR, CAPTIONS_DIR, External_Captions,  INSTANCE_NAME, Session_Name, OUTPUT_DIR, ofstnselvl, Save_VRAM):


    if os.path.exists(INSTANCE_DIR+"/.ipynb_checkpoints"):
      call('rm -r '+INSTANCE_DIR+'/.ipynb_checkpoints', shell=True)
    if os.path.exists(CAPTIONS_DIR+"/.ipynb_checkpoints"):
      call('rm -r '+CAPTIONS_DIR+'/.ipynb_checkpoints', shell=True)


    Seed=random.randint(1, 999999)
    
    ofstnse=""
    if Offset_Noise:
      ofstnse="--offset_noise"
      
    GC=''
    if Save_VRAM:
        GC='--gradient_checkpointing'
        
    extrnlcptn=""
    if External_Captions:
      extrnlcptn="--external_captions"      

    precision="fp16"

   
   
    def train_only_text(SESSION_DIR, MODEL_NAME, INSTANCE_DIR, OUTPUT_DIR, Seed, Resolution, ofstnse, extrnlcptn, precision, Training_Epochs):
        print('[1;33mTraining the Text Encoder...[0m')
        call('accelerate launch /workspace/diffusers/examples/dreambooth/train_dreambooth_rnpd_sdxl_TI.py \
        '+ofstnse+' \
        '+extrnlcptn+' \
        --dim='+str(dim)+' \
        --ofstnselvl='+str(ofstnselvl)+' \
        --image_captions_filename \
        --Session_dir='+SESSION_DIR+' \
        --pretrained_model_name_or_path='+MODEL_NAME+' \
        --instance_data_dir='+INSTANCE_DIR+' \
        --output_dir='+OUTPUT_DIR+' \
        --captions_dir='+CAPTIONS_DIR+' \
        --seed='+str(Seed)+' \
        --resolution='+str(Resolution)+' \
        --mixed_precision='+str(precision)+' \
        --train_batch_size=1 \
        --gradient_accumulation_steps=1 '+GC+ ' \
        --use_8bit_adam \
        --learning_rate='+str(Text_Encoder_Learning_Rate)+' \
        --lr_scheduler="cosine" \
        --lr_warmup_steps=0 \
        --num_train_epochs='+str(Training_Epochs), shell=True)
   
   

    def train_only_unet(SESSION_DIR, MODEL_NAME, INSTANCE_DIR, OUTPUT_DIR, Seed, Resolution, ofstnse, extrnlcptn, precision, Training_Epochs):
        print('[1;33mTraining the UNet...[0m')
        call('accelerate launch /workspace/diffusers/examples/dreambooth/train_dreambooth_rnpd_sdxl_lora.py \
        '+ofstnse+' \
        '+extrnlcptn+' \
        --dim='+str(dim)+' \
        --ofstnselvl='+str(ofstnselvl)+' \
        --image_captions_filename \
        --Session_dir='+SESSION_DIR+' \
        --pretrained_model_name_or_path='+MODEL_NAME+' \
        --instance_data_dir='+INSTANCE_DIR+' \
        --output_dir='+OUTPUT_DIR+' \
        --captions_dir='+CAPTIONS_DIR+' \
        --seed='+str(Seed)+' \
        --resolution='+str(Resolution)+' \
        --mixed_precision='+str(precision)+' \
        --train_batch_size=1 \
        --gradient_accumulation_steps=1 '+GC+ ' \
        --use_8bit_adam \
        --learning_rate='+str(Unet_Learning_Rate)+' \
        --lr_scheduler="cosine" \
        --lr_warmup_steps=0 \
        --num_train_epochs='+str(Training_Epochs), shell=True)

      

    if Unet_Training_Epochs!=0:
        if Text_Encoder_Training_Epochs!=0:
            train_only_text(SESSION_DIR, MODEL_NAME, INSTANCE_DIR, OUTPUT_DIR, Seed, Resolution, ofstnse, extrnlcptn, precision, Training_Epochs=Text_Encoder_Training_Epochs)
            clear_output()
        train_only_unet(SESSION_DIR, MODEL_NAME, INSTANCE_DIR, OUTPUT_DIR, Seed, Resolution, ofstnse, extrnlcptn, precision, Training_Epochs=Unet_Training_Epochs)
    else  :
      print('[1;32mNothing to do')


    if os.path.exists(SESSION_DIR+'/'+Session_Name+'.safetensors'):
        clear_output()
        print("[1;32mDONE, the LoRa model is in the session's folder")
    else:
        print("[1;31mSomething went wrong")




def sdcmff(Huggingface_token_optional, MDLPTH, restored):

    from slugify import slugify
    from huggingface_hub import HfApi, CommitOperationAdd, create_repo
    
    os.chdir('/workspace')

    if restored:
        Huggingface_token_optional=""

    if Huggingface_token_optional!="":
       username = HfApi().whoami(Huggingface_token_optional)["name"]
       backup=f"https://huggingface.co/datasets/{username}/fast-stable-diffusion/resolve/main/sdcomfy_backup_rnpd.tar.zst"
       headers = {"Authorization": f"Bearer {Huggingface_token_optional}"}
       response = requests.head(backup, headers=headers)
       if response.status_code == 302:
          restored=True
          print('[1;33mRestoring ComfyUI...')
          open('/workspace/sdcomfy_backup_rnpd.tar.zst', 'wb').write(requests.get(backup, headers=headers).content)
          call('tar --zstd -xf sdcomfy_backup_rnpd.tar.zst', shell=True)
          call('rm sdcomfy_backup_rnpd.tar.zst', shell=True)
       else:
          print('[1;33mBackup not found, using a fresh/existing repo...')
          time.sleep(2)
          if not os.path.exists('ComfyUI'):
              call('git clone -q --depth 1 https://github.com/comfyanonymous/ComfyUI', shell=True)
    else:
        print('[1;33mInstalling/Updating the repo...')
        if not os.path.exists('ComfyUI'):
            call('git clone -q --depth 1 https://github.com/comfyanonymous/ComfyUI', shell=True)

    os.chdir('ComfyUI')
    call('git reset --hard', shell=True)
    print('[1;32m')
    call('git pull', shell=True)

    if os.path.exists(MDLPTH):
        call('ln -s '+MDLPTH+' models/loras', shell=True, stdout=open('/dev/null', 'w'), stderr=open('/dev/null', 'w'))
        
    clean_symlinks('models/loras')        

    if not os.path.exists('models/checkpoints/sd_xl_base_1.0.safetensors'):
        call('cp /workspace/auto-models/sd_xl_base_1.0.safetensors models/checkpoints', shell=True)


    podid=os.environ.get('RUNPOD_POD_ID')
    localurl=f"https://{podid}-3001.proxy.runpod.net"
    call("sed -i 's@print(\"To see the GUI go to: http://{}:{}\".format(address, port))@print(\"[32m\u2714 Connected\")\\n            print(\"[1;34m"+localurl+"[0m\")@' /workspace/ComfyUI/server.py", shell=True)
    os.chdir('/workspace')
    
    return restored 




def test(MDLPTH, User, Password, Huggingface_token_optional, restoreda):

    from slugify import slugify
    from huggingface_hub import HfApi, CommitOperationAdd, create_repo
    import gradio
    
    gradio.close_all()
               

    auth=f"--gradio-auth {User}:{Password}"
    if User =="" or Password=="":
      auth=""


    if restoreda:
        Huggingface_token_optional=""

    if Huggingface_token_optional!="":
       username = HfApi().whoami(Huggingface_token_optional)["name"]
       backup=f"https://huggingface.co/datasets/{username}/fast-stable-diffusion/resolve/main/sd_backup_rnpd.tar.zst"
       headers = {"Authorization": f"Bearer {Huggingface_token_optional}"}
       response = requests.head(backup, headers=headers)
       if response.status_code == 302:
          restoreda=True
          print('[1;33mRestoring the SD folder...')
          open('/workspace/sd_backup_rnpd.tar.zst', 'wb').write(requests.get(backup, headers=headers).content)
          call('tar --zstd -xf sd_backup_rnpd.tar.zst', shell=True)
          call('rm sd_backup_rnpd.tar.zst', shell=True)
       else:
          print('[1;33mBackup not found, using a fresh/existing repo...')
          time.sleep(2)
          if not os.path.exists('/workspace/sd/stablediffusiond'): #reset later
             call('wget -q -O sd_mrep.tar.zst https://huggingface.co/TheLastBen/dependencies/resolve/main/sd_mrep.tar.zst', shell=True)
             call('tar --zstd -xf sd_mrep.tar.zst', shell=True)
             call('rm sd_mrep.tar.zst', shell=True)        
          os.chdir('/workspace/sd')
          if not os.path.exists('stable-diffusion-webui'):
              call('git clone -q --depth 1 --branch master https://github.com/AUTOMATIC1111/stable-diffusion-webui', shell=True)            
        
    else:
        print('[1;33mInstalling/Updating the repo...')
        os.chdir('/workspace')
        if not os.path.exists('/workspace/sd/stablediffusiond'): #reset later
           call('wget -q -O sd_mrep.tar.zst https://huggingface.co/TheLastBen/dependencies/resolve/main/sd_mrep.tar.zst', shell=True)
           call('tar --zstd -xf sd_mrep.tar.zst', shell=True)
           call('rm sd_mrep.tar.zst', shell=True)        

        os.chdir('/workspace/sd')
        if not os.path.exists('stable-diffusion-webui'):
            call('git clone -q --depth 1 --branch master https://github.com/AUTOMATIC1111/stable-diffusion-webui', shell=True)


    os.chdir('/workspace/sd/stable-diffusion-webui/')
    call('git reset --hard', shell=True)
    print('[1;32m')
    call('git pull', shell=True)
    
    
    if os.path.exists(MDLPTH):
        call('mkdir models/Lora', shell=True, stdout=open('/dev/null', 'w'), stderr=open('/dev/null', 'w'))
        call('ln -s '+MDLPTH+' models/Lora', shell=True, stdout=open('/dev/null', 'w'), stderr=open('/dev/null', 'w'))
    
    if not os.path.exists('models/Stable-diffusion/sd_xl_base_1.0.safetensors'):
        call('cp /workspace/auto-models/sd_xl_base_1.0.safetensors models/Stable-diffusion', shell=True)

    clean_symlinks('models/Lora')
   
    os.chdir('/workspace')


    call('wget -q -O /usr/local/lib/python3.10/dist-packages/gradio/blocks.py https://raw.githubusercontent.com/TheLastBen/fast-stable-diffusion/main/AUTOMATIC1111_files/blocks.py', shell=True)
   
    os.chdir('/workspace/sd/stable-diffusion-webui/modules')
    
    call("sed -i 's@possible_sd_paths =.*@possible_sd_paths = [\"/workspace/sd/stablediffusion\"]@' /workspace/sd/stable-diffusion-webui/modules/paths.py", shell=True)
    call("sed -i 's@\.\.\/@src/@g' /workspace/sd/stable-diffusion-webui/modules/paths.py", shell=True)
    call("sed -i 's@src\/generative-models@generative-models@g' /workspace/sd/stable-diffusion-webui/modules/paths.py", shell=True)
    
    call("sed -i 's@\[\"sd_model_checkpoint\"\]@\[\"sd_model_checkpoint\", \"sd_vae\", \"CLIP_stop_at_last_layers\", \"inpainting_mask_weight\", \"initial_noise_multiplier\"\]@g' /workspace/sd/stable-diffusion-webui/modules/shared.py", shell=True)
    call("sed -i 's@print(\"No module.*@@' /workspace/sd/stablediffusion/ldm/modules/diffusionmodules/model.py", shell=True)
    os.chdir('/workspace/sd/stable-diffusion-webui')
    clear_output()

    podid=os.environ.get('RUNPOD_POD_ID')
    localurl=f"{podid}-3001.proxy.runpod.net"

    for line in fileinput.input('/usr/local/lib/python3.10/dist-packages/gradio/blocks.py', inplace=True):
      if line.strip().startswith('self.server_name ='):
          line = f'            self.server_name = "{localurl}"\n'
      if line.strip().startswith('self.protocol = "https"'):
          line = '            self.protocol = "https"\n'
      if line.strip().startswith('if self.local_url.startswith("https") or self.is_colab'):
          line = ''
      if line.strip().startswith('else "http"'):
          line = ''
      sys.stdout.write(line)


    configf="--disable-console-progressbars --upcast-sampling --no-half-vae --disable-safe-unpickle --api --opt-sdp-attention --enable-insecure-extension-access --no-download-sd-model  --skip-version-check  --listen --port 3000 --ckpt /workspace/sd/stable-diffusion-webui/models/Stable-diffusion/sd_xl_base_1.0.safetensors "+auth
    
    
    return configf, restoreda




def clean():
    
    Sessions=os.listdir("/workspace/Fast-Dreambooth/Sessions")

    s = widgets.Select(
        options=Sessions,
        rows=5,
        description='',
        disabled=False
    )

    out=widgets.Output()

    d = widgets.Button(
        description='Remove',
        disabled=False,
        button_style='warning',
        tooltip='Removet the selected session',
        icon='warning'
    )

    def rem(d):
        with out:
            if s.value is not None:
                clear_output()
                print("[1;33mTHE SESSION [1;31m"+s.value+" [1;33mHAS BEEN REMOVED FROM THE STORAGE")
                call('rm -r /workspace/Fast-Dreambooth/Sessions/'+s.value, shell=True)
                if os.path.exists('/workspace/models/'+s.value):
                  call('rm -r /workspace/models/'+s.value, shell=True)
                s.options=os.listdir("/workspace/Fast-Dreambooth/Sessions")       


            else:
                d.close()
                s.close()
                clear_output()
                print("[1;32mNOTHING TO REMOVE")

    d.on_click(rem)
    if s.value is not None:
        display(s,d,out)
    else:
        print("[1;32mNOTHING TO REMOVE")



def crop_image(im, size):

  import cv2

  GREEN = "#0F0"
  BLUE = "#00F"
  RED = "#F00"    

  def focal_point(im, settings):
      corner_points = image_corner_points(im, settings) if settings.corner_points_weight > 0 else []
      entropy_points = image_entropy_points(im, settings) if settings.entropy_points_weight > 0 else []
      face_points = image_face_points(im, settings) if settings.face_points_weight > 0 else []

      pois = []

      weight_pref_total = 0
      if len(corner_points) > 0:
        weight_pref_total += settings.corner_points_weight
      if len(entropy_points) > 0:
        weight_pref_total += settings.entropy_points_weight
      if len(face_points) > 0:
        weight_pref_total += settings.face_points_weight

      corner_centroid = None
      if len(corner_points) > 0:
        corner_centroid = centroid(corner_points)
        corner_centroid.weight = settings.corner_points_weight / weight_pref_total 
        pois.append(corner_centroid)

      entropy_centroid = None
      if len(entropy_points) > 0:
        entropy_centroid = centroid(entropy_points)
        entropy_centroid.weight = settings.entropy_points_weight / weight_pref_total
        pois.append(entropy_centroid)

      face_centroid = None
      if len(face_points) > 0:
        face_centroid = centroid(face_points)
        face_centroid.weight = settings.face_points_weight / weight_pref_total 
        pois.append(face_centroid)

      average_point = poi_average(pois, settings)
      
      return average_point


  def image_face_points(im, settings):

      np_im = np.array(im)
      gray = cv2.cvtColor(np_im, cv2.COLOR_BGR2GRAY)

      tries = [
        [ f'{cv2.data.haarcascades}haarcascade_eye.xml', 0.01 ],
        [ f'{cv2.data.haarcascades}haarcascade_frontalface_default.xml', 0.05 ],
        [ f'{cv2.data.haarcascades}haarcascade_profileface.xml', 0.05 ],
        [ f'{cv2.data.haarcascades}haarcascade_frontalface_alt.xml', 0.05 ],
        [ f'{cv2.data.haarcascades}haarcascade_frontalface_alt2.xml', 0.05 ],
        [ f'{cv2.data.haarcascades}haarcascade_frontalface_alt_tree.xml', 0.05 ],
        [ f'{cv2.data.haarcascades}haarcascade_eye_tree_eyeglasses.xml', 0.05 ],
        [ f'{cv2.data.haarcascades}haarcascade_upperbody.xml', 0.05 ]
      ]
      for t in tries:
        classifier = cv2.CascadeClassifier(t[0])
        minsize = int(min(im.width, im.height) * t[1]) # at least N percent of the smallest side
        try:
          faces = classifier.detectMultiScale(gray, scaleFactor=1.1,
            minNeighbors=7, minSize=(minsize, minsize), flags=cv2.CASCADE_SCALE_IMAGE)
        except:
          continue

        if len(faces) > 0:
          rects = [[f[0], f[1], f[0] + f[2], f[1] + f[3]] for f in faces]
          return [PointOfInterest((r[0] +r[2]) // 2, (r[1] + r[3]) // 2, size=abs(r[0]-r[2]), weight=1/len(rects)) for r in rects]
      return []


  def image_corner_points(im, settings):
      grayscale = im.convert("L")

      # naive attempt at preventing focal points from collecting at watermarks near the bottom
      gd = ImageDraw.Draw(grayscale)
      gd.rectangle([0, im.height*.9, im.width, im.height], fill="#999")

      np_im = np.array(grayscale)

      points = cv2.goodFeaturesToTrack(
          np_im,
          maxCorners=100,
          qualityLevel=0.04,
          minDistance=min(grayscale.width, grayscale.height)*0.06,
          useHarrisDetector=False,
      )

      if points is None:
          return []

      focal_points = []
      for point in points:
        x, y = point.ravel()
        focal_points.append(PointOfInterest(x, y, size=4, weight=1/len(points)))

      return focal_points


  def image_entropy_points(im, settings):
      landscape = im.height < im.width
      portrait = im.height > im.width
      if landscape:
        move_idx = [0, 2]
        move_max = im.size[0]
      elif portrait:
        move_idx = [1, 3]
        move_max = im.size[1]
      else:
        return []

      e_max = 0
      crop_current = [0, 0, settings.crop_width, settings.crop_height]
      crop_best = crop_current
      while crop_current[move_idx[1]] < move_max:
          crop = im.crop(tuple(crop_current))
          e = image_entropy(crop)

          if (e > e_max):
            e_max = e
            crop_best = list(crop_current)

          crop_current[move_idx[0]] += 4
          crop_current[move_idx[1]] += 4

      x_mid = int(crop_best[0] + settings.crop_width/2)
      y_mid = int(crop_best[1] + settings.crop_height/2)

      return [PointOfInterest(x_mid, y_mid, size=25, weight=1.0)]


  def image_entropy(im):
      # greyscale image entropy
      # band = np.asarray(im.convert("L"))
      band = np.asarray(im.convert("1"), dtype=np.uint8)
      hist, _ = np.histogram(band, bins=range(0, 256))
      hist = hist[hist > 0]
      return -np.log2(hist / hist.sum()).sum()

  def centroid(pois):
    x = [poi.x for poi in pois]
    y = [poi.y for poi in pois]
    return PointOfInterest(sum(x)/len(pois), sum(y)/len(pois))


  def poi_average(pois, settings):
      weight = 0.0
      x = 0.0
      y = 0.0
      for poi in pois:
          weight += poi.weight
          x += poi.x * poi.weight
          y += poi.y * poi.weight
      avg_x = round(weight and x / weight)
      avg_y = round(weight and y / weight)

      return PointOfInterest(avg_x, avg_y)


  def is_landscape(w, h):
    return w > h


  def is_portrait(w, h):
    return h > w


  def is_square(w, h):
    return w == h


  class PointOfInterest:
    def __init__(self, x, y, weight=1.0, size=10):
      self.x = x
      self.y = y
      self.weight = weight
      self.size = size

    def bounding(self, size):
      return [
        self.x - size//2,
        self.y - size//2,
        self.x + size//2,
        self.y + size//2
      ]

  class Settings:
    def __init__(self, crop_width=512, crop_height=512, corner_points_weight=0.5, entropy_points_weight=0.5, face_points_weight=0.5):
      self.crop_width = crop_width
      self.crop_height = crop_height
      self.corner_points_weight = corner_points_weight
      self.entropy_points_weight = entropy_points_weight
      self.face_points_weight = face_points_weight

  settings = Settings(
      crop_width = size,
      crop_height = size,
      face_points_weight = 0.9,
      entropy_points_weight = 0.15,
      corner_points_weight = 0.5,
  )        

  scale_by = 1
  if is_landscape(im.width, im.height):
    scale_by = settings.crop_height / im.height
  elif is_portrait(im.width, im.height):
    scale_by = settings.crop_width / im.width
  elif is_square(im.width, im.height):
    if is_square(settings.crop_width, settings.crop_height):
      scale_by = settings.crop_width / im.width
    elif is_landscape(settings.crop_width, settings.crop_height):
      scale_by = settings.crop_width / im.width
    elif is_portrait(settings.crop_width, settings.crop_height):
      scale_by = settings.crop_height / im.height

  im = im.resize((int(im.width * scale_by), int(im.height * scale_by)))
  im_debug = im.copy()

  focus = focal_point(im_debug, settings)

  # take the focal point and turn it into crop coordinates that try to center over the focal
  # point but then get adjusted back into the frame
  y_half = int(settings.crop_height / 2)
  x_half = int(settings.crop_width / 2)

  x1 = focus.x - x_half
  if x1 < 0:
      x1 = 0
  elif x1 + settings.crop_width > im.width:
      x1 = im.width - settings.crop_width

  y1 = focus.y - y_half
  if y1 < 0:
      y1 = 0
  elif y1 + settings.crop_height > im.height:
      y1 = im.height - settings.crop_height

  x2 = x1 + settings.crop_width
  y2 = y1 + settings.crop_height

  crop = [x1, y1, x2, y2]

  results = []

  results.append(im.crop(tuple(crop)))

  return results
  


def clean_symlinks(path):
    for item in os.listdir(path):
        lnk = os.path.join(path, item)
        if os.path.islink(lnk) and not os.path.exists(os.readlink(lnk)):
            os.remove(lnk)
  
