from models.efficientnet import EfficientNet
if __name__ == '__main__':
    model = EfficientNet.from_pretrained('efficientnet-b0')
    inputs = torch.randn(4, 3, 640, 640)
    P = model(inputs)
    for idx, p in enumerate(P):
        print('P{}: {}'.format(idx, p.size()))
    # print('model: ', model)
