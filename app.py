import streamlit as st
from diffusers import DPMSolverMultistepScheduler
from PIL import Image, ImageEnhance
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
import torch


SAVE_LOCATION = 'prompt.jpg'

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
        
    return grid


def main():
    st.set_page_config(layout="wide")
    st.title("AppNation Simple Playground")
    
    image = Image.open('1_cropped.jpeg')
    st.image(image, caption='One of the used reference images')
    # Create text prompt
    prompt = st.text_input('Input the prompt desired. Please use "<teddy-bear>" identifier.')
    inference_steps = st.text_input('Please type the number of inference steps. Increasing the number of steps will also increase the processing time.')
    st.markdown(f"""
        This demo is using a free CPU. It will be slow.
    """)
    out_image = None
    
    if st.button('Generate'):
    # if len(prompt) > 0:
        st.markdown(f"""
            This will show an image using **stable diffusion** of the desired {prompt} entered:
        """)
        print(prompt)
        # Create a spinner to show the image is being generated
        with st.spinner('Generating image based on prompt'):
            try:
                pipe
            except NameError:
                pipe = StableDiffusionPipeline.from_pretrained(
                    'dreambooth-concept',
                    scheduler = DPMSolverMultistepScheduler.from_pretrained('dreambooth-concept', subfolder="scheduler"),
                    torch_dtype=torch.float32,
                )
                
            # prompt = "a \u003Cteddy-bear> as an actor in front of hollywood sign" 
            image = pipe(prompt, num_images_per_prompt=1, num_inference_steps=int(inference_steps), guidance_scale=9).images
            st.success('Generated image.')
            out_image = image_grid(image, 1, 1)

    # Open and display the image on the site
    # image = Image.open(SAVE_LOCATION)
        st.image(out_image)
    
    
if __name__ == '__main__':
    main()
