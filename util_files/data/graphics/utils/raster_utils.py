import base64
from io import BytesIO

from PIL import Image
import numpy as np


def image_to_datauri(image, format='jpeg'):
    image = Image.fromarray(np.asarray(image))
    buffer = BytesIO()
    if format == 'jpeg':
        image.save(buffer, format=format, quality=95)
        encoding = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f'data:image/jpeg;base64,{encoding}'
    else:
        raise NotImplementedError(f'image_to_datauri is not implemented for {format} format')
