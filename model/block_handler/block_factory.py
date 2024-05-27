import model.block_handler.block as block

class BlockFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_block(name, latent_dim, in_feat, out_feat, normalize=True):
        if name == 'basic':
            return block.Basic(latent_dim, in_feat, out_feat, normalize=True)
