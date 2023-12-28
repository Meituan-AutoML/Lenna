import torch
import torch.nn as nn

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM, LlavaLlamaModel)

from mmdet.models.data_preprocessors.data_preprocessor import DetDataPreprocessor
from .grounding_dino.build_gdino_mmdet import build_gdino


class LennaMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LennaMetaModel, self).__init__(config)

        self.config = config
        self.vision_pretrained = kwargs.get("vision_pretrained", None)
        self.initialize_lenna_modules(self.config)

    def initialize_lenna_modules(self, config):
        self.visual_model = build_gdino()
        for param in self.visual_model.parameters():
            param.requires_grad = False
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

class LennaModel(LennaMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LennaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LennaForCausalLM(LlavaLlamaForCausalLM):
    def __init__(self, config, **kwargs, ):
        super().__init__(config)
        self.det_token_idx = kwargs.pop("det_token_idx")
        self.model = LennaModel(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self.data_preprocessor = DetDataPreprocessor(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_mask=False,
        )
       
    def evaluate(self, images_clip, dino_target, input_ids, resize_list, original_size_list, max_new_tokens=32, tokenizer=None, caption=None, ):
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences
            det_token_mask = output_ids[:, 1:] == self.det_token_idx
            det_token_mask = torch.cat([torch.zeros((det_token_mask.shape[0], 255)).bool().cuda(), det_token_mask,], dim=1, )
            hidden_states = []
            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))
            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[det_token_mask]

            det_token_counts = det_token_mask.int().sum(-1)
            det_token_offset = det_token_counts.cumsum(-1)
            det_token_offset = torch.cat([torch.zeros(1).long().cuda(), det_token_offset], dim=0)

            pred_embeddings_ = []
            for i in range(len(det_token_offset) - 1):
                start_i, end_i = det_token_offset[i], det_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_
            this_device = images_clip.device
            all_dino_inputs = torch.stack([d['inputs'] for d in [dino_target]]).to(this_device)
            all_dino_data = [d['data_samples'] for d in [dino_target]]
            dino_target_new = {
                'inputs': all_dino_inputs,
                'data_samples': all_dino_data,
            }
            dino_inputs = self.data_preprocessor(dino_target_new)
            pred_embeddings = torch.stack(pred_embeddings).to(this_device)
            pred_text_token_mask = torch.tensor([[True]]).to(this_device)
            dino_inputs['data_samples'][0] = dino_inputs['data_samples'][0].to('cuda')
            dino_inputs['data_samples'][0].gt_instances.bboxes = dino_inputs['data_samples'][0].gt_instances.bboxes.type(torch.float32)
            dino_inputs['data_samples'][0].ignored_instances.bboxes = dino_inputs['data_samples'][0].ignored_instances.bboxes.type(torch.float32)
            dino_output = self.model.visual_model.predict(dino_inputs['inputs'].to(this_device), dino_inputs['data_samples'], pred_embeddings, pred_text_token_mask)

        return output_ids, dino_output
