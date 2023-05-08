import argparse
import cv2
import os
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
from pathlib import Path
import gradio as gr
import matplotlib.pyplot as plt
from loguru import logger
import subprocess
import copy
import time
import warnings

import torch
from torchvision.ops import box_convert
warnings.filterwarnings("ignore")

# grounding DINO
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, load_image, predict
import groundingdino.datasets.transforms as T

# segment anything
from segment_anything import build_sam, SamPredictor 

#stable diffusion
from diffusers import StableDiffusionInpaintPipeline

from huggingface_hub import hf_hub_download

if not os.path.exists('./demo1.jpg'):
    os.system("wget https://github.com/IDEA-Research/Grounded-Segment-Anything/raw/main/assets/demo1.jpg")

if not os.path.exists('./sam_vit_h_4b8939.pth'):
    logger.info(f"get sam_vit_h_4b8939.pth...")
    result = subprocess.run(['wget', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'], check=True)
    print(f'wget sam_vit_h_4b8939.pth result = {result}')

# Use this command for evaluate the GLIP-T model
config_file = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filename = "groundingdino_swint_ogc.pth"
sam_checkpoint = './sam_vit_h_4b8939.pth' 
output_dir = "outputs"
groundingdino_device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'device={device}')

# make dir
os.makedirs(output_dir, exist_ok=True)


def load_model_hf(model_config_path, repo_id, filename, device='cpu'):
    args = SLConfig.fromfile(model_config_path) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model    

def load_image_and_transform(init_image):
    init_image = init_image.convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image, _ = transform(init_image, None) # 3, h, w
    return init_image, image

def image_transform_grounding_for_vis(init_image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
    ])
    image, _ = transform(init_image, None) # 3, h, w
    return image

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        font = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
        font_size = 20
        new_font = ImageFont.truetype(font, font_size)

        draw.text((x0+2, y0+2), str(label), font=new_font, fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0,0,0,0), lw=1)) 
    ax.text(x0, y0+20, label, fontdict={'fontsize': 6}, color="white")

def get_grounding_box(image_tensor, grounding_caption, box_threshold, text_threshold):
    # run grounding
    boxes, logits, phrases = predict(groundingDino_model, image_tensor, grounding_caption, box_threshold, text_threshold, device=groundingdino_device)
    labels = [
        f"{phrase} ({logit:.2f})"
        for phrase, logit
        in zip(phrases, logits)
    ]
    # annotated_frame = annotate(image_source=np.asarray(image_pil), boxes=boxes, logits=logits, phrases=phrases)
    # image_with_box = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    return boxes, labels

def mask_extend(img, box, extend_pixels=10, useRectangle=True):
    box[0] = int(box[0])
    box[1] = int(box[1])
    box[2] = int(box[2])
    box[3] = int(box[3])
    region = img.crop(tuple(box))                           # crop based on bb box
    new_width = box[2] - box[0] + 2*extend_pixels           
    new_height = box[3] - box[1] + 2*extend_pixels

    region_BILINEAR = region.resize((int(new_width), int(new_height)))   # resize the cropped region based on "extend_pixels"
    if useRectangle:
        region_draw = ImageDraw.Draw(region_BILINEAR)
        region_draw.rectangle((0, 0, new_width, new_height), fill=(255, 255, 255))       # draw white rectangle
    img.paste(region_BILINEAR, (int(box[0]-extend_pixels), int(box[1]-extend_pixels)))   #pastes the resized region back into the original image at the same location as the original bounding box but with an additional padding of extend_pixels pixels on all sides
    return img

def mix_masks(imgs):
    re_img =  1 - np.asarray(imgs[0].convert("1"))
    for i in range(len(imgs)-1):
        re_img = np.multiply(re_img, 1 - np.asarray(imgs[i+1].convert("1")))
    re_img =  1 - re_img
    return  Image.fromarray(np.uint8(255*re_img))

def run_anything_task(input_image, text_prompt, task_type, inpaint_prompt, box_threshold, text_threshold, 
            iou_threshold, inpaint_mode, mask_source_radio, remove_mode, remove_mask_extend):

    text_prompt = text_prompt.strip()

    # user guidance messages
    if not (task_type == 'inpainting' or task_type == 'remove'):
        if text_prompt == '':
            return [], gr.Gallery.update(label='Please input detection prompt~~')
    
    if input_image is None:
            return [], gr.Gallery.update(label='Please upload a image~~')
    
    file_temp = int(time.time())

    # load mask
    input_mask_pil = input_image['mask']
    input_mask = np.array(input_mask_pil.convert("L"))  

    # load image
    image_pil, image_tensor = load_image_and_transform(input_image['image'])

    # RUN GROUNDINGDINO: we skip DINO if we draw mask on the image
    if (task_type == 'inpainting' or task_type == 'remove') and mask_source_radio == mask_source_draw:
        pass
    else:
        boxes, phrases = get_grounding_box(image_tensor, text_prompt, box_threshold, text_threshold)
        if boxes.size(0) == 0:
                logger.info(f'run_grounded_sam_[]_{task_type}_[{text_prompt}]_1_[No objects detected, please try others.]_')
                return [], gr.Gallery.update(label='No objects detected, please try others!')
        boxes_filt_ori = copy.deepcopy(boxes)

        size = image_pil.size
        
        pred_dict = {
                "boxes": boxes,
                "size": [size[1], size[0]],  # H,W
                "labels": phrases,
            }

        # store and save DINO output
        output_images = []
        image_with_box = plot_boxes_to_image(copy.deepcopy(image_pil), pred_dict)[0]
        image_path = os.path.join(output_dir, f"grounding_dino_output_{file_temp}.jpg")
        image_with_box.save(image_path)
        detection_image_result = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        os.remove(image_path)
        output_images.append(detection_image_result)

    # if mask is detected from DINO
    logger.info(f'run_anything_task_[{file_temp}]_{task_type}_2_')
    if task_type == 'segment' or ((task_type == 'inpainting' or task_type == 'remove') 
                                and mask_source_radio == mask_source_segment):
        image = np.array(input_image['image'])
        sam_predictor.set_image(image)
    
        # map the bounding boxes from dino to original size
        h, w = size[1], size[0]
        boxes = boxes * torch.Tensor([w, h, w, h])
        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
        # can use box_convert function or below
        # for i in range(boxes.size(0)):
        #     boxes[i] = boxes[i] * torch.Tensor([W, H, W, H])
        #     boxes[i][:2] -= boxes[i][2:] / 2   # top left corner
        #     boxes[i][2:] += boxes[i][:2]       # bottom left corner

        # transform boxes from original ratio to sam's zoomed ratio 
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes, image.shape[:2])

        # predict masks/segmentation
        # masks: [number of masks, C, H, W] but note that H and W is 512
        masks, _, _ = sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
        )

        # draw output image
        plt.figure()
        plt.imshow(image)
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes, phrases):
            show_box(box.numpy(), plt.gca(), label)
        plt.axis('off')
        image_path = os.path.join(output_dir, f"grounding_seg_output_{file_temp}.jpg")
        plt.savefig(image_path, bbox_inches="tight")
        segment_image_result = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        os.remove(image_path)
        output_images.append(segment_image_result)
    
    logger.info(f'run_anything_task_[{file_temp}]_{task_type}_3_')
    if task_type == 'segment':
        logger.info(f'run_anything_task_[{file_temp}]_{task_type}_Final_')
        return output_images, gr.Gallery.update(label='result images')

    elif task_type == 'inpainting' or task_type == 'remove':
        # if no inpaint prompt is entered, we treat it as remove
        if inpaint_prompt.strip() == '' and mask_source_radio == mask_source_segment:
            task_type = 'remove'

        logger.info(f'run_anything_task_[{file_temp}]_{task_type}_4_')  
        if mask_source_radio == mask_source_draw:
            mask_pil = input_mask_pil
            mask = input_mask          
        else:
            masks_ori = copy.deepcopy(masks)
            # inpainting pipeline
            if inpaint_mode == 'merge':
                masks = torch.sum(masks, dim=0).unsqueeze(0)
                masks = torch.where(masks > 0, True, False)

            # simply choose the first mask, which will be refine in the future release
            mask = masks[0][0].cpu().numpy()
            mask_pil = Image.fromarray(mask)   
        output_images.append(mask_pil.convert("RGB"))

        if task_type == 'inpainting':
            # inpainting pipeline
            image_source_for_inpaint = image_pil.resize((512, 512))
            image_mask_for_inpaint = mask_pil.resize((512, 512))
            image_inpainting = sd_pipe(prompt=inpaint_prompt, image=image_source_for_inpaint, mask_image=image_mask_for_inpaint).images[0]            
        # else: add remove option here!!

        image_inpainting = image_inpainting.resize((image_pil.size[0], image_pil.size[1]))
        output_images.append(image_inpainting)
        return output_images, gr.Gallery.update(label='result images')        
    else:
        logger.info(f"task_type:{task_type} error!")
    logger.info(f'run_anything_task_[{file_temp}]_Final_Inpainting_')
    return output_images, gr.Gallery.update(label='result images')


def change_radio_display(task_type, mask_source_radio):
    text_prompt_visible = True
    inpaint_prompt_visible = False
    mask_source_radio_visible = False

    if task_type == "inpainting":
        inpaint_prompt_visible = True
    if task_type == "inpainting" or task_type == "remove":
        mask_source_radio_visible = True   
        if mask_source_radio == mask_source_draw:
            text_prompt_visible = False

    return  gr.Textbox.update(visible=text_prompt_visible), gr.Textbox.update(visible=inpaint_prompt_visible), gr.Radio.update(visible=mask_source_radio_visible) 



# model initialization
groundingDino_model = load_model_hf(config_file, ckpt_repo_id, ckpt_filename, groundingdino_device)
sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint))

# initialize stable-diffusion-inpainting
logger.info(f"initialize stable-diffusion-inpainting...")
sd_pipe = None
if os.environ.get('IS_MY_DEBUG') is None:
    sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", 
            torch_dtype=torch.float16
    )
    sd_pipe = sd_pipe.to(device)

if __name__ == "__main__":

    mask_source_draw = "Draw mask on image."
    mask_source_segment = "Segment based on prompt and inpaint."

    parser = argparse.ArgumentParser("Grounding SAM demo", add_help=True)
    parser.add_argument("--debug", action="store_true", help="using debug mode")
    parser.add_argument("--share", action="store_true", help="share the app")
    args = parser.parse_args()

    print(f'args = {args}')

    block = gr.Blocks().queue()
    with block:
        gr.Markdown("# GroundingDino SAM and Stable Diffusion")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    source="upload", elem_id="image_upload", type="pil", tool="sketch", value="demo1.jpg", label="Upload")
                task_type = gr.Radio(["segment", "inpainting", "remove"],  value="segment", 
                                                label='Task type', visible=True)
                
                mask_source_radio = gr.Radio([mask_source_draw, mask_source_segment], 
                                    value=mask_source_segment, label="Mask from",
                                    visible=False) 
                
                text_prompt = gr.Textbox(label="Detection Prompt, seperating each name with dot '.', i.e.: bear.cat.dog.chair ]", \
                                         value='bear', placeholder="Cannot be empty")                                                
                inpaint_prompt = gr.Textbox(label="Inpaint Prompt (if this is empty, then remove)", visible=False)
                
                run_button = gr.Button(label="Run")
                with gr.Accordion("Advanced options", open=False):
                    box_threshold = gr.Slider(
                        label="Box Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001
                    )
                    text_threshold = gr.Slider(
                        label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001
                    )
                    iou_threshold = gr.Slider(
                        label="IOU Threshold", minimum=0.0, maximum=1.0, value=0.8, step=0.001
                    )
                    inpaint_mode = gr.Radio(["merge", "first"], value="merge", label="inpaint_mode")
                    with gr.Row():
                        with gr.Column(scale=1):
                            remove_mode = gr.Radio(["segment", "rectangle"],  value="segment", label='remove mode') 
                        with gr.Column(scale=1):
                            remove_mask_extend = gr.Textbox(label="remove_mask_extend", value='10')

            with gr.Column():
                gallery = gr.Gallery(label="result images", show_label=True, elem_id="gallery", visible=True
                ).style(preview=True, columns=[5], object_fit="scale-down", height="auto")

        task_type.change(fn=change_radio_display, inputs=[task_type, mask_source_radio], outputs=[text_prompt, inpaint_prompt, mask_source_radio])
        mask_source_radio.change(fn=change_radio_display, inputs=[task_type, mask_source_radio], outputs=[text_prompt, inpaint_prompt, mask_source_radio])
        
        DESCRIPTION = '### This demo from [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) and kudos to thier excellent works. Welcome everyone to try this out and learn together!'
        gr.Markdown(DESCRIPTION)

        run_button.click(fn=run_anything_task, inputs=[
                        input_image, text_prompt, task_type, inpaint_prompt,
                        box_threshold,text_threshold, iou_threshold, inpaint_mode,
                        mask_source_radio, remove_mode, remove_mask_extend], 
                        outputs=[gallery, gallery], show_progress=True, queue=True)

    block.launch(debug=args.debug, share=args.share, show_api=False, show_error=True)