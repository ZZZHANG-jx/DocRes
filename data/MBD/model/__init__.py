import torchvision.models as models
from model.densenetccnl import *
from model.unetnc import *
from model.gienet import *


def get_model(name, n_classes=1, filters=64,version=None,in_channels=3, is_batchnorm=True, norm='batch', model_path=None, use_sigmoid=True, layers=3,img_size=512):
    model = _get_model_instance(name)


    if name == 'dnetccnl':
        model = model(img_size=128, in_channels=in_channels, out_channels=n_classes, filters=32)
    elif name == 'dnetccnl512':
        model = model(img_size=img_size, in_channels=in_channels, out_channels=n_classes, filters=32)
    elif name == 'unetnc':
        model = model(input_nc=in_channels, output_nc=n_classes, num_downs=7)
    elif name == 'gie':
        model = model(input_nc=in_channels, output_nc=n_classes, num_downs=7)
    elif name == 'giecbam':
        model = model(input_nc=in_channels, output_nc=n_classes, num_downs=7)
    elif name == 'gie2head':
        model = model(input_nc=in_channels, output_nc=n_classes, num_downs=7)
    elif name == 'giemask':
        model = model(input_nc=in_channels, output_nc=n_classes, num_downs=7)
    elif name == 'giemask2':
        model = model(input_nc=in_channels, output_nc=n_classes, num_downs=7)
    elif name == 'giedilated':
        model = model(input_nc=in_channels, output_nc=n_classes, num_downs=7)
    elif name == 'bmp':
        model = model(input_nc=in_channels, output_nc=n_classes, num_downs=7)
    elif name == 'displacement':
        model = model(n_classes=2, num_filter=32, BatchNorm='GN', in_channels=5)
    return model

def _get_model_instance(name):
    try:
        return {
            'dnetccnl': dnetccnl,
            'dnetccnl512': dnetccnl512,
            'unetnc': UnetGenerator,
            'gie':GieGenerator,
            'giecbam':GiecbamGenerator,
            'giedilated':DilatedSingleUnet,
            'gie2head':Gie2headGenerator,
            'giemask':GiemaskGenerator,
            'giemask2':Giemask2Generator,
            'bmp':BmpGenerator,
        }[name]
    except:
        print('Model {} not available'.format(name))
