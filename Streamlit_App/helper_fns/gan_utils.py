import os
from PIL import Image

def make_square(image, min_size=512, fill_color=(255, 255, 255, 0)):
    '''
        Make a square image with signature in the center and black (transparent)
        strips on top and bottom. Cycle GAN is trained with images of this format.    
    '''
    x, y = image.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(image, (int((size - x) / 2), int((size - y) / 2)))
    new_im = new_im.resize((512, 512))
    return new_im

def resize_images(path):
    '''
        Resize all the images present in path that matches the ips used in cyclegan
        training
    '''
    dirs = os.listdir(path)
    for item in dirs:
        if os.path.isfile(path+item):
            image = Image.open(path+item)
            image = make_square(image)
            image = image.convert('RGB')
            image.save(path+item)

# test_path = 'runs/detect/exp/crops/DLSignature/'
# resize_images(test_path)