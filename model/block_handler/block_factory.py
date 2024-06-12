import model.block_handler.block as block

class BlockFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_block(name, in_feat, out_feat, normalize = False, last = False):
        if name == 'basic':
            blocks = block.Basic(in_feat, out_feat, normalize, last)
            return blocks.layers
        
        elif name == 'Residual':
            return [block.Residual(in_feat, out_feat, normalize)]
        
        elif name == 'transformer':
            blocks = block.TransformerBlock(in_feat, out_feat, normalize, last)
            return [blocks]

        elif name == 'VGG':
            blocks = block.VGG(in_feat, out_feat, normalize, last)
            return blocks.layers
        
        elif name == 'Resnet':
            blocks = block.ResNet(in_feat, out_feat, normalize)
            return [blocks]
        
        elif name == 'Dense':
            growth_rate = 32  
            num_layers = 1
            blocks = block.Dense(num_layers, in_feat, growth_rate, out_feat, normalize)
            return [blocks]
    
        elif name == 'InceptionV1':
            blocks = block.InceptionV1(in_feat, out_feat, normalize, last)
            return [blocks]

        elif name == 'InceptionV2':
            blocks = block.InceptionV2(in_feat, out_feat, normalize, last)
            return [blocks]
        
        elif name == 'DCNv3':
            blocks = block.DCNv3(in_feat, out_feat, normalize, last)
            return [blocks]
        

        