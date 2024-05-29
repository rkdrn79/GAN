import model.block_handler.block as block

class BlockFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_block(name, in_feat, out_feat, normalize=False):
        if name == 'basic':
            blocks = block.Basic(in_feat, out_feat, normalize)
            return blocks.layers
