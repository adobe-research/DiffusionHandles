# import boto3
# import botocore
import cv2
import distutils
import hashlib
import io
import itertools
import json
import numpy
import os
import PIL
import PIL.Image
import PIL.ImageCms
import random
# import s3fs
import shutil
import ssl
import struct
import time
import torch
import tqdm
import typing
import urllib
import urllib.request
import math



def bgr_to_gray(tenIn:torch.Tensor):
    if type(tenIn) == numpy.ndarray:
        return bgr_to_gray(torch.FloatTensor(tenIn.transpose(2, 0, 1)[None, :, :, :])).numpy()[0, :, :, :].transpose(1, 2, 0)
    # end

    assert(tenIn.ndim == 4 and tenIn.shape[1] == 3 and tenIn.dtype in [torch.float16, torch.float32, torch.float64])

    tenB = tenIn[:, 0:1, :, :]
    tenG = tenIn[:, 1:2, :, :]
    tenR = tenIn[:, 2:3, :, :]

    return (0.114 * tenB) + (0.587 * tenG) + (0.299 * tenR)


def depth_to_points(tenIn:typing.Union[torch.Tensor, numpy.ndarray], fltFov:float, fltPrincipal:typing.List[float]=[0.5, 0.5]):
    if type(tenIn) == numpy.ndarray:
        return depth_to_points(torch.FloatTensor(tenIn.transpose(2, 0, 1)[None, :, :, :]), fltFov, fltPrincipal).numpy()[0, :, :, :].transpose(1, 2, 0)
    # end

    assert(tenIn.ndim == 4 and tenIn.shape[1] == 1 and tenIn.dtype in [torch.float16, torch.float32, torch.float64])
    assert(fltFov > 0.0)

    fltFocal = 0.5 * max(tenIn.shape[3], tenIn.shape[2]) * math.tan(math.radians(90.0) - (0.5 * math.radians(fltFov)))

    tenX = torch.linspace(start=((0.0 - fltPrincipal[0]) * tenIn.shape[3]) + 0.5, end=((1.0 - fltPrincipal[0]) * tenIn.shape[3]) - 0.5, steps=tenIn.shape[3], dtype=tenIn.dtype, device=tenIn.device).view(1, 1, 1, tenIn.shape[3]).repeat(tenIn.shape[0], 1, tenIn.shape[2], 1)
    tenY = torch.linspace(start=((0.0 - fltPrincipal[1]) * tenIn.shape[2]) + 0.5, end=((1.0 - fltPrincipal[1]) * tenIn.shape[2]) - 0.5, steps=tenIn.shape[2], dtype=tenIn.dtype, device=tenIn.device).view(1, 1, tenIn.shape[2], 1).repeat(tenIn.shape[0], 1, 1, tenIn.shape[3])

    return torch.cat([tenX * (1.0 / fltFocal) * tenIn, tenY * (1.0 / fltFocal) * tenIn, tenIn], 1)


def read_binary(strFile:str, **kwargs):
    assert(type(strFile) == str)

    if 'objClient' not in kwargs: kwargs['objClient'] = None

    assert(all([strArg in ['objClient'] for strArg in kwargs]) == True)
    assert(kwargs['objClient'] is None or isinstance(kwargs['objClient'], botocore.client.BaseClient) == True)

    strFile = strFile.replace('//', '/').replace('//', '/').replace('//', '/').replace(':/', '://')

    if strFile.startswith('http') == True:
        objContext = ssl.create_default_context()

        objContext.check_hostname = False
        objContext.verify_mode = ssl.CERT_NONE

        with urllib.request.urlopen(
            url=urllib.request.Request(
                url=strFile,
                headers={
                    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0' # https://github.com/rom1504/img2dataset/blob/main/img2dataset/downloader.py
                }
            ),
            context=objContext,
            timeout=60
        ) as objFile:
            strBinary = objFile.read()
        # end

    elif strFile.startswith('s3') == True:
        if kwargs['objClient'] is None:
            if 'boto3' not in objIocache:
                objIocache['boto3'] = {
                    'authed': boto3.session.Session().client(service_name='s3', config=botocore.config.Config(region_name='us-west-2', connect_timeout=10, read_timeout=10, retries={'mode': 'standard', 'max_attempts': 10})),
                    'anon': boto3.session.Session().client(service_name='s3', config=botocore.config.Config(region_name='us-west-2', connect_timeout=10, read_timeout=10, retries={'mode': 'standard', 'max_attempts': 10}, signature_version=botocore.UNSIGNED))
                }
            # end

            kwargs['objClient'] = objIocache['boto3']['anon']
        # end

        strBucket = strFile.replace('s3://', '').split('/')[0]
        strKey = str('/').join(strFile.replace('s3://', '').split('/')[1:])

        strBinary = kwargs['objClient'].get_object(Bucket=strBucket, Key=strKey)['Body'].read()

    elif True:
        with open(strFile, 'rb') as objFile:
            strBinary = objFile.read()
        # end

    # end

    return strBinary
# end


def read_image(strFile:str, **kwargs):
    assert(type(strFile) == str)

    if 'boolApplyicc' not in kwargs: kwargs['boolApplyicc'] = False
    if 'boolGrayscale' not in kwargs: kwargs['boolGrayscale'] = False
    if 'boolAlpha' not in kwargs: kwargs['boolAlpha'] = False
    if 'boolFloatify' not in kwargs: kwargs['boolFloatify'] = True
    if 'objClient' not in kwargs: kwargs['objClient'] = None

    assert(all([strArg in ['boolApplyicc', 'boolGrayscale', 'boolAlpha', 'boolFloatify', 'objClient'] for strArg in kwargs]) == True)
    assert(type(kwargs['boolApplyicc']) == bool)
    assert(type(kwargs['boolGrayscale']) == bool)
    assert(type(kwargs['boolAlpha']) == bool)
    assert(type(kwargs['boolFloatify']) == bool)
    assert(kwargs['objClient'] is None or isinstance(kwargs['objClient'], botocore.client.BaseClient) == True)

    strImage = read_binary(strFile, **{strKey: kwargs[strKey] for strKey in kwargs if strKey in ['objClient']})

    if strImage[:8] == ('REDIRECT').encode('utf-8'):
        return read_image(strImage.decode('utf-8').splitlines()[1], **kwargs)
    # end

    if strFile.endswith('.pfm') == True:
        strImage = io.BytesIO(strImage)

        strMeta = strImage.readline().decode('utf-8').strip()
        intChans = {'Pf': 1, 'PF': 3}[strMeta]

        strMeta = strImage.readline().decode('utf-8').strip()
        intWidth = int(strMeta.split(' ')[0])
        intHeight = int(strMeta.split(' ')[1])

        strMeta = strImage.readline().decode('utf-8').strip()
        strEndian = '<' if float(strMeta) < 0.0 else '>'
        fltScale = abs(float(strMeta)) # monkaa doesn't need the scale and middlestereo seems to be wrong if we use it

        npyImage = numpy.ascontiguousarray(numpy.frombuffer(strImage.read(), strEndian + 'f').reshape(intHeight, intWidth, intChans)[::-1, :, :])

    elif strImage[:4].hex() == '716f6966':
        npyImage = qoi.decode(strImage) # forgot to swap the color channels so the qoi better be in bgr format

    elif strImage[:4].hex() == '762f3101':
        npyImage = cv2.imdecode(buf=numpy.frombuffer(strImage, numpy.uint8), flags=-1)

        if npyImage is None:
            write_binary('/dev/shm/tuls-' + hashlib.md5(strImage).hexdigest() + '-' + str(os.getpid()), strImage)

            # sudo apt-get install libopenexr-dev && pip install git+https://github.com/jamesbowman/openexrpython.git && pip install git+https://github.com/tvogels/pyexr.git

            npyImage = __import__('pyexr').read('/dev/shm/tuls-' + hashlib.md5(strImage).hexdigest() + '-' + str(os.getpid()))

            file_remove('/dev/shm/tuls-' + hashlib.md5(strImage).hexdigest() + '-' + str(os.getpid()))
        # end

    elif kwargs['boolApplyicc'] == True:
        try:
            objImage = PIL.Image.open(io.BytesIO(strImage))
        except:
            return read_image(strFile, **{**kwargs, 'boolApplyicc': False})
        # end

        if objImage.mode not in ['RGB', 'RGBA']:
            return read_image(strFile, **{**kwargs, 'boolApplyicc': False})
        # end

        npyAlpha = numpy.array(objImage)[:, :, 3:4] if objImage.mode == 'RGBA' else None

        try:
            objImage = PIL.ImageCms.applyTransform(objImage, PIL.ImageCms.buildTransform(PIL.ImageCms.ImageCmsProfile(io.BytesIO(objImage.info.get('icc_profile'))), PIL.ImageCms.createProfile('sRGB'), 'RGB', 'RGB'))
        except:
            pass
        # end

        npyImage = numpy.array(objImage)

        if npyAlpha is not None and npyImage.ndim == 3 and npyImage.shape[2] == 3:
            npyImage = numpy.concatenate([npyImage, npyAlpha], 2)
        # end

        if npyImage.ndim == 3 and npyImage.shape[2] == 3:
            npyImage = npyImage[:, :, ::-1]

        elif npyImage.ndim == 3 and npyImage.shape[2] == 4:
            npyImage = npyImage[:, :, [2, 1, 0, 3]]

        # end

    elif True:
        npyImage = cv2.imdecode(buf=numpy.frombuffer(strImage, numpy.uint8), flags=-1)

    # end

    if npyImage is None:
        return None
    # end

    if npyImage.ndim != 3:
        npyImage = numpy.atleast_3d(npyImage)
    # end

    if kwargs['boolGrayscale'] == False:
        if npyImage.shape[2] == 1:
            if strFile.split('.')[-1].lower() in ['bmp', 'gif', 'jpg', 'jpeg', 'png', 'qoi', 'webp']:
                npyImage = npyImage.repeat(3, 2)
            # end
        # end
    # end

    if kwargs['boolAlpha'] == False:
        npyImage = npyImage[:, :, 0:3]
    # end

    if kwargs['boolFloatify'] == True:
        if npyImage.dtype == numpy.uint8:
            npyImage = npyImage.astype(numpy.float32) * (1.0 / 255.0)

        elif npyImage.dtype == numpy.uint16:
            npyImage = npyImage.astype(numpy.float32) * (1.0 / 65535.0)

        # end
    # end

    return npyImage
# end

