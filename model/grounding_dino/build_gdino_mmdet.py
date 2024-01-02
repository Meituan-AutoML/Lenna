from .grounding_dino import GroundingDINO
from .grounding_dino_head import LennaGroundingDINOHead

def build_gdino():
    lang_model_name = 'bert-base-uncased'
    model = GroundingDINO(
        num_queries=900,
        with_box_refine=True,
        as_two_stage=True,
        language_model=dict(
            type='BertModel',
            name=lang_model_name,
            pad_to_max=False,
            use_sub_sentence_represent=True,
            special_tokens_list=['[CLS]', '[SEP]', '.', '?'],
            add_pooling_layer=False,
        ),
        backbone=dict(
            type='SwinTransformer',
            pretrain_img_size=384,
            embed_dims=192,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=12,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            with_cp=False,
            convert_weights=True,
            ),
        neck=dict(
            type='ChannelMapper',
            in_channels=[192, 384, 768, 1536],
            kernel_size=1,
            out_channels=256,
            act_cfg=None,
            bias=True,
            norm_cfg=dict(type='GN', num_groups=32),
            num_outs=4),
        encoder=dict(
            num_layers=6,
            num_cp=6,
            # visual layer config
            layer_cfg=dict(
                self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
                ffn_cfg=dict(
                    embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
            # text layer config
            text_layer_cfg=dict(
                self_attn_cfg=dict(num_heads=4, embed_dims=256, dropout=0.0),
                ffn_cfg=dict(
                    embed_dims=256, feedforward_channels=1024, ffn_drop=0.0)),
            # fusion layer config
            fusion_layer_cfg=dict(
                v_dim=256,
                l_dim=256,
                embed_dim=1024,
                num_heads=4,
                init_values=1e-4),
        ),
        decoder=dict(
            num_layers=6,
            return_intermediate=True,
            layer_cfg=dict(
                # query self attention layer
                self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
                # cross attention layer query to text
                cross_attn_text_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
                # cross attention layer query to image
                cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
                ffn_cfg=dict(
                    embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
            post_norm_cfg=None),
        positional_encoding=dict(
            num_feats=128, normalize=True, offset=0.0, temperature=20),
        bbox_head=dict(
            type='LennaGroundingDINOHead',
            num_classes=365,
            sync_cls_avg_factor=True,
            contrastive_cfg=dict(max_text_len=256, log_scale='auto', bias=True),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),  # 2.0 in DeformDETR
            loss_bbox=dict(type='L1Loss', loss_weight=5.0),
            loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
        dn_cfg=dict(  # TODO: Move to model.train_cfg ?
            label_noise_scale=0.5,
            box_noise_scale=1.0,  # 0.4 for DN-DETR
            group_cfg=dict(dynamic=True, num_groups=None,
                        num_dn_queries=100)),  # TODO: half num_dn_queries
        # training and testing settings
        train_cfg=dict(
            assigner=dict(
                type='HungarianAssigner',
                match_costs=[
                    dict(type='BinaryFocalLossCost', weight=2.0),
                    dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                    dict(type='IoUCost', iou_mode='giou', weight=2.0)
                ])),
        test_cfg=dict(max_per_img=300)
    )
    return model
