
# image to base64 converter
import base64
from io import BytesIO
from PIL import Image
from typing import Union

d = {
    "python": 0.534,
    "java": 0.123,
    "c": 0.223,
    "c++": 0.33,
    "c#": 0.11,
    "javascript": 0.44,
    "php": 0.22,
    "html": 0.33,
}







import matplotlib.pyplot as plt
from wordcloud import WordCloud

#wordcloud = WordCloud(background_color="white").generate_from_frequencies(d)


# function that takes matplotlib canvas and returns base64 string

def image_to_base64(figure: Union[plt.Figure, WordCloud]) -> str:
    """
    Convert matplotlib figure or WordCloud to base64 string.

    Args:
        figure: Matplotlib figure or WordCloud object

    Returns:
        str: Base64 encoded string of the image
    """
    buffer = BytesIO()

    if isinstance(figure, WordCloud):
        figure.to_image().save(buffer, format='PNG')
    else:
        figure.savefig(buffer, format='PNG', bbox_inches='tight', pad_inches=0)

    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()

    return image_base64


# Example usage
#print(image_to_base64(wordcloud))


def base64_to_image(base64_string: str) -> Image.Image:
    """
    Convert base64 string to PIL Image.

    Args:
        base64_string: Base64 encoded string of the image

    Returns:
        Image: PIL Image object
    """
    img_data = base64.b64decode(base64_string)
    buffer = BytesIO(img_data)
    image = Image.open(buffer)
    # Load the image data into memory before closing buffer
    image.load()
    buffer.close()

    return image
# Example usage
#base64_string = image_to_base64(wordcloud)
#
#image = base64_to_image(base64_string)
##save image
#image.save("wordcloud.png")
