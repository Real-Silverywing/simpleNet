# from models.testnet import SATestNet
# from models.mix_attn_net import MixAttnNet
from models.simple_net import SimpleNet, SimpleNet_cnn
# from models.simple_attn_net import SimpleAttnNet
# from models.temporal_attn_net import TempAttnNet
# from models.temporal_spatial_attn_net import SpatTempAttnNet
# from models.simple_traj_net import SimpleTrajNet
# from models.interpretable_temporal_spatial_attn_net import (
#     InterpretableSpatTempAttnNet,
# )
# from models.temporal_spatial_attn_traj_net import TrajSpatTempAttnNet
# from models.feature_temp_net import FeatureTemporalNetwork


# def testnet(config, drop_rate=0):
#     return SATestNet()
#
#
# def mixattnnet(config, drop_rate=0):
#     return MixAttnNet(num_classes=1, drop_rate=drop_rate)


def simplenet(config, drop_rate=0):
    return SimpleNet(num_classes=1, drop_rate=drop_rate)

def simplenet_cnn(config, drop_rate=0):
    return SimpleNet_cnn(num_classes=1, drop_rate=drop_rate)


# def simpleattnnet(config, drop_rate=0):
#     return SimpleAttnNet(num_classes=1, drop_rate=drop_rate)
#
#
# def tempattnnet(config, drop_rate=0):
#     return TempAttnNet(num_classes=1, drop_rate=drop_rate)
#
#
# def tempspatattnnet(config, drop_rate=0, attnvis=False):
#     return SpatTempAttnNet(
#         num_classes=config.num_classes, drop_rate=drop_rate, attnvis=attnvis
#     )
#
#
# def intertempspatattnnet(config, drop_rate=0, attnvis=False):
#     print("DEBUG intertempspatattnnet: use_rgb_base: ", config.use_rgb_base)
#     return InterpretableSpatTempAttnNet(
#         num_classes=config.num_classes,
#         drop_rate=drop_rate,
#         attnvis=attnvis,
#         use_rgb_base=config.use_rgb_base,
#         out_attention=config.attention_loss,
#         select=config.select,
#         decoder_dim=config.decoder_dim,
#         temporal_attention=config.temporal_attention,
#         use_spatial_attention=config.spatial_attn,
#         spatial_attn_pool=config.spatial_attn_pool,
#         unfreeze_rgb=config.unfreeze_rgb,
#         decoder=config.decoder,
#         vid_length=config.clip_size*config.nclips,
#     )
#
#
# def trajtempspatattnnet(config, drop_rate=0, attnvis=False):
#     print("DEBUG trajtempspatattnnet: use_rgb_base: ", config.use_rgb_base)
#     return TrajSpatTempAttnNet(
#         num_classes=config.num_classes,
#         drop_rate=drop_rate,
#         attnvis=attnvis,
#         use_rgb_base=config.use_rgb_base,
#         out_attention=config.trajectory,
#         select=config.select,
#     )
#
#
# def simpletrajnet(config, drop_rate=0):
#     return SimpleTrajNet(num_classes=1, drop_rate=drop_rate)
#
#
# def feattempnetwork(config, drop_rate=0):
#     return FeatureTemporalNetwork(
#         config.decoder_dim,
#         config.model_type,
#         temporal_attention=config.temporal_attention,
#         temporal_attention_softmax=config.temporal_attention_softmax,
#     )

