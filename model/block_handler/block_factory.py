import model.block_handler.block as block

class BlockFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_block(name, in_feat, out_feat, normalize = False, last = False):
        if name == 'basic':
            blocks = block.Basic(in_feat, out_feat, normalize, last)
            return blocks.layers
        
        elif name == 'transformer':
            blocks = block.TransformerBlock(in_feat, out_feat, nhead = 1, num_layers = 1)
            return blocks.layers

        elif name == 'cnn':
            blocks = block.CNN(in_feat, out_feat, normalize, last)
