from PIL import Image


def concat_image(images, gap=0):
    """
    Concatenate multiple images into one function

    :param images: List of PIL images
    :param gap: gap between each image
    :return: One image which is concatenated of all images
    """
    #
    if len(images) == 1:
        return images[0]

    #
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths) + (len(images) - 1) * gap
    max_height = max(heights)

    #
    new_im = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0] + gap

    return new_im