from PIL import Image, ImageDraw
import numpy as np

# convert, supported modes: "L", "RGBX", "RGBA", and "CMYK"
image_src = Image.open('JackMa.jpg').convert('RGBA')
print image_src.size
width, height = image_src.size  # size of the source image

# prepare a mask in image size
image_mask = np.zeros(shape=(height, width, 3)).astype(np.uint8)    # reverse order of W and H for array
image_mask[(height/4):(height/4*3), (width/4):(width/4*3), :] = (255, 20, 147)  # mask with deep pink
# Hot Pink	    255-105-180	ff69b4
# Deep Pink	    255-20-147	ff1493
# Pink	        255-192-203	ffc0cb
# Light Pink	255-182-193	ffb6c1
image_mask = Image.fromarray(image_mask).convert('RGBA')    # image_mask.getbands() = 'RGB'
# # to draw on image, e.g. add text, not in use in this case
# image_draw = ImageDraw.ImageDraw(image_mask, 'RGBA')    # create with alpha layer
# to fade the mask by alpha filter
#   see <http://www.pythonware.com/library/pil/handbook/image.htm#Image.point>
mask_alpha = image_mask.convert('L').point(lambda x: min(x, 100))     # mask.getbands() = 'L'

# use alpha filter to make it transparent
image_mask.putalpha(mask_alpha)
image_src.paste(image_mask, None, image_mask)
image_src.save('JackMa_drawmask.jpg', "JPEG")
