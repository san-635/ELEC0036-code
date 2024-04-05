from cgitb import text
import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt.modules.vision_transformer as vit

from vilt.modules import heads, objectives, vilt_utils

class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters() # save to self.hparams
        
        # text embedding module
        self.text_embeddings = heads.EHREmbedding(config["ehr_n_var"], config["hidden_size"])       ### modified
        self.text_embeddings.apply(objectives.init_weights)

        # modality-type embedding module
        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        # set up ViT model
        if self.hparams.config["load_path"] == "":  # load vit_base_patch32_384 with pretrained weighs if no load_path is provided
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:                                       # load vit_base_patch32_384 with random weighs if load_path is provided
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )
        # self.transformer is an instance of VisionTransformer class from vision_transformer.py

        # pooling layer
        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)

        # different heads for different pre-training tasks - IGNORE
        if config["loss_names"]["mlm"] > 0: # masked language modeling
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0: # image-text matching
            self.itm_score = heads.ITMHead(config["hidden_size"])
            self.itm_score.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0: # masked patch prediction
            self.mpp_score = heads.MPPHead(bert_config)
            self.mpp_score.apply(objectives.init_weights)
            
        if config["loss_names"]["mppd"] > 0:
            self.mppd_score = heads.MPPHead(bert_config)
            self.mppd_score.apply(objectives.init_weights)

        # ===================== Downstream tasks ======================= #
        # load pretrained ViLT model's weights into vit_base_patch32_384 model given load_path is provided
        if (self.hparams.config["load_path"] != "" and not self.hparams.config["finetune_first"] and not self.hparams.config["test_only"]):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu") # load checkpoint file at load_path
            state_dict = ckpt["state_dict"]
            if config["max_text_len"] != 40:    # replace position embeddings weights with interpolated ones if max_text_len != 40
                state_dict['text_embeddings.position_ids'] = torch.Tensor(range(config["max_text_len"])).long().view(1,-1)
                pos_emb = state_dict['text_embeddings.position_embeddings.weight']
                pos_emb = torch.nn.functional.interpolate(pos_emb.view(1,1,40,768), size=(config["max_text_len"],768), mode='bilinear').squeeze()
                state_dict['text_embeddings.position_embeddings.weight'] = pos_emb
            self.load_state_dict(state_dict, strict=False)

        # task-specific classifiers (sequences of layers) depending on the datasets
        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["vqa"] > 0:        # for Visual Question Answering (VQA)
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["ehr_cxr"] > 0:    # for EHR+CXR dataset (binary classification)     ### added
            cls_num = self.hparams.config["class_num"]  # 2
            self.ehr_cxr_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, cls_num),
            )
            self.ehr_cxr_classifier.apply(objectives.init_weights)
            
        if self.hparams.config["loss_names"]["hatememes"] > 0:  # for Hateful Memes (binary classification)
            cls_num = self.hparams.config["hatememes_class_num"]
            self.hatememes_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, cls_num),
            )
            self.hatememes_classifier.apply(objectives.init_weights)
            
        if self.hparams.config["loss_names"]["food101"] > 0:    # for Food101 (multi-class classification)
            cls_num = self.hparams.config["food101_class_num"]
            self.food101_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, cls_num),
            )
            self.food101_classifier.apply(objectives.init_weights)   
            
        if self.hparams.config["loss_names"]["mmimdb"] > 0:     # for MM-IMDb (multi-label classification)
            cls_num = self.hparams.config["mmimdb_class_num"]
            self.mmimdb_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, cls_num),
            )
            self.mmimdb_classifier.apply(objectives.init_weights)          
            
            if self.hparams.config["load_path"] != "" and self.hparams.config["finetune_first"]:
                ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
                state_dict = ckpt["state_dict"]
                self.load_state_dict(state_dict, strict=False)
                print("use pre-finetune model")
            self.missing_ratio = self.hparams.config["test_ratio"]
            self.exp_name = self.hparams.config["test_exp_name"]
            self.test_type = self.hparams.config["test_type"]

            if self.hparams.config['fix_model']:    # freeze ViLT backbone by stopping optimiser from updating parameters
                print('fix ViLT backbone')
                for param in self.transformer.parameters():
                    param.requires_grad=False
                for param in self.text_embeddings.parameters():
                    param.requires_grad=False
                for param in self.token_type_embeddings.parameters():
                    param.requires_grad=False                

        if self.hparams.config["loss_names"]["nlvr2"] > 0:      # for Natural Language for Visual Reasoning 2 (NLVR2)
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        if self.hparams.config["loss_names"]["irtr"] > 0:    # for Flickr30k and MSCOCO (image retrieval and text retrieval tasks)
            self.rank_output = nn.Linear(hs, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_score.parameters():
                p.requires_grad = False

        # define metrics for each task (e.g. VQA score for VQA, F1 score for MM-IMDb, AUROC for Hateful Memes, etc.)
        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== Downstream tasks (if config["test_only"] == True) ===================== #
        # load pretrained ViLT model's weights into vit_base_patch32_384 model given load_path is provided
        # no consideration of max_text_len != 40 case now
        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

    # ===================== Inference ===================== #
    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
        is_train=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"    # imgkey = "image_0"
        else:
            imgkey = "image"                                # imgkey = "image"

        # image embeddings - pass image data through ViT's visual_embed module to get image_embeds, etc.
        if image_embeds is None and image_masks is None:
            img = batch[imgkey][0]                          # shape: batch_size, 3, H, W
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,   # None since mask_it=False
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
            )
        else:
            patch_index, image_labels = (
                None,
                None,
            )
        
        # text embeddings - pass EHR data through EHREmbeddings module to get text_embeds
        text_embeds = self.text_embeddings(batch["text"])           # shape: batch_size, max_text_len-1, ehr_n_var -> batch_size, max_text_len, hidden_size     ### added

        # add modal-type embeddings (tensors of 0s for text, 1s for image) to text and image embeddings; shapes are unchanged
        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_embeds[:, :, 0]).long()),               ### modified
            image_embeds + self.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx))
        )

        ### added
        # prepend cls_token to masks of EHR data
        text_masks = batch["mask"]                                                    # shape: batch_size, max_text_len-1   ### added
        cls_token = torch.zeros((text_masks.size(0), 1), device=text_masks.device)
        text_masks = torch.cat([cls_token, text_masks], dim=1)                        # shape: batch_size, max_text_len

        # concatenate text and image embeddings, and text and image masks (by columns)
        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)       # shape: batch_size, max_text_len + {image_embeds.size(1)}, hidden_size
        co_masks = torch.cat([text_masks, image_masks], dim=1)          # shape: batch_size, max_text_len + {image_masks.size(1)}

        # pass concatenated embeddings and masks through transformer
        x = co_embeds
        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)    # call each Block class instance's forward() method

        # normalise the concatenated transformer output
        x = self.transformer.norm(x)

        # separate text and image features from transformer output
        text_feats, image_feats = (x[:, : text_embeds.shape[1]], x[:, text_embeds.shape[1] :])

        # pass the cls_token from transformer output through pooling layer to get cls_feats
        cls_feats = self.pooler(x)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_labels_mppd": image_feats,
            "image_masks": image_masks,
            "text_masks": text_masks,
            "patch_index": patch_index,
        }
        return ret

    # ===================== Forward pass ===================== #
    def forward(self, batch):
        ret = dict()

        # if current_tasks is empty, return the output of infer()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret
        
        # call corresponding compute_ function from objectives.py for each task in current_tasks
        # this function calls infer() to get the embeddings and masks, and then passes them through the corresponding task-specific classifier
        # then, the output of the classifier is passed through the corresponding loss function to get the loss
        # finally, the loss is added to the ret dict

        # Masked Language Modeling objective
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Masked Patch Prediction objective
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))
            
        # Masked Patch Prediction objective
        if "mppd" in self.current_tasks:
            ret.update(objectives.compute_mppd(self, batch))

        # Image Text Matching objective
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_wpa(self, batch))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))
            
        # Binary classification for EHR+CXR dataset     ### added
        if "ehr_cxr" in self.current_tasks:
            ret.update(objectives.compute_ehr_cxr(self, batch))
        
        # Binary classification for Hateful Memes
        if "hatememes" in self.current_tasks:
            ret.update(objectives.compute_hatememes(self, batch))
            
        # Classification for Food101
        if "food101" in self.current_tasks:
            ret.update(objectives.compute_food101(self, batch))        
            
        # Multi-label classification for MM-IMDb
        if "mmimdb" in self.current_tasks:
            ret.update(objectives.compute_mmimdb(self, batch))
            
        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        # Image Retrieval and Text Retrieval for Flickr30k and MSCOCO
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)       # set current_tasks to the tasks in config["loss_names"] with non-zero values, e.g. ["mmimdb", "itm"]
        output = self(batch)            # call forward() with batch as input argument
        total_loss = sum([v for k, v in output.items() if "loss" in k])  # sum up all losses (from different tasks if there are multiple)
        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)   # compute and log epoch-level metrics, ie accuracy_epoch, AUROC_epoch, loss_epoch, the_metric

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)   # compute and log epoch-level metrics, ie accuracy_epoch, AUROC_epoch, loss_epoch, the_metric

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)            # call forward() with batch as input argument
        ret = dict()
        return ret                      # but return an empty dict since it's a test step

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)
    
class ViLTransformerSS_EHR(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters() # save to self.hparams
        
        # text embedding module
        self.text_embeddings = heads.EHREmbedding(config["ehr_n_var"], config["hidden_size"])       ### modified
        self.text_embeddings.apply(objectives.init_weights)

        # modality-type embedding module
        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        # set up ViT model
        if self.hparams.config["load_path"] == "":  # load vit_base_patch32_384 with pretrained weighs if no load_path is provided
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:                                       # load vit_base_patch32_384 with random weighs if load_path is provided
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )
        # self.transformer is an instance of VisionTransformer class from vision_transformer.py

        # pooling layer
        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)

        # different heads for different pre-training tasks - IGNORE
        if config["loss_names"]["mlm"] > 0: # masked language modeling
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0: # image-text matching
            self.itm_score = heads.ITMHead(config["hidden_size"])
            self.itm_score.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0: # masked patch prediction
            self.mpp_score = heads.MPPHead(bert_config)
            self.mpp_score.apply(objectives.init_weights)
            
        if config["loss_names"]["mppd"] > 0:
            self.mppd_score = heads.MPPHead(bert_config)
            self.mppd_score.apply(objectives.init_weights)

        # ===================== Downstream tasks ======================= #
        # load pretrained ViLT model's weights into vit_base_patch32_384 model given load_path is provided
        if (self.hparams.config["load_path"] != "" and not self.hparams.config["finetune_first"] and not self.hparams.config["test_only"]):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu") # load checkpoint file at load_path
            state_dict = ckpt["state_dict"]
            if config["max_text_len"] != 40:    # replace position embeddings weights with interpolated ones if max_text_len != 40
                state_dict['text_embeddings.position_ids'] = torch.Tensor(range(config["max_text_len"])).long().view(1,-1)
                pos_emb = state_dict['text_embeddings.position_embeddings.weight']
                pos_emb = torch.nn.functional.interpolate(pos_emb.view(1,1,40,768), size=(config["max_text_len"],768), mode='bilinear').squeeze()
                state_dict['text_embeddings.position_embeddings.weight'] = pos_emb
            self.load_state_dict(state_dict, strict=False)

        # task-specific classifiers (sequences of layers) depending on the datasets
        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["vqa"] > 0:        # for Visual Question Answering (VQA)
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["ehr_cxr"] > 0:    # for EHR+CXR dataset (binary classification)     ### added
            cls_num = self.hparams.config["class_num"]  # 2
            self.ehr_cxr_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, cls_num),
            )
            self.ehr_cxr_classifier.apply(objectives.init_weights)
            
        if self.hparams.config["loss_names"]["hatememes"] > 0:  # for Hateful Memes (binary classification)
            cls_num = self.hparams.config["hatememes_class_num"]
            self.hatememes_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, cls_num),
            )
            self.hatememes_classifier.apply(objectives.init_weights)
            
        if self.hparams.config["loss_names"]["food101"] > 0:    # for Food101 (multi-class classification)
            cls_num = self.hparams.config["food101_class_num"]
            self.food101_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, cls_num),
            )
            self.food101_classifier.apply(objectives.init_weights)   
            
        if self.hparams.config["loss_names"]["mmimdb"] > 0:     # for MM-IMDb (multi-label classification)
            cls_num = self.hparams.config["mmimdb_class_num"]
            self.mmimdb_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, cls_num),
            )
            self.mmimdb_classifier.apply(objectives.init_weights)          
            
            if self.hparams.config["load_path"] != "" and self.hparams.config["finetune_first"]:
                ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
                state_dict = ckpt["state_dict"]
                self.load_state_dict(state_dict, strict=False)
                print("use pre-finetune model")
            self.missing_ratio = self.hparams.config["test_ratio"]
            self.exp_name = self.hparams.config["test_exp_name"]
            self.test_type = self.hparams.config["test_type"]

            if self.hparams.config['fix_model']:    # freeze ViLT backbone by stopping optimiser from updating parameters
                print('fix ViLT backbone')
                for param in self.transformer.parameters():
                    param.requires_grad=False
                for param in self.text_embeddings.parameters():
                    param.requires_grad=False
                for param in self.token_type_embeddings.parameters():
                    param.requires_grad=False                

        if self.hparams.config["loss_names"]["nlvr2"] > 0:      # for Natural Language for Visual Reasoning 2 (NLVR2)
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        if self.hparams.config["loss_names"]["irtr"] > 0:    # for Flickr30k and MSCOCO (image retrieval and text retrieval tasks)
            self.rank_output = nn.Linear(hs, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_score.parameters():
                p.requires_grad = False

        # define metrics for each task (e.g. VQA score for VQA, F1 score for MM-IMDb, AUROC for Hateful Memes, etc.)
        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== Downstream tasks (if config["test_only"] == True) ===================== #
        # load pretrained ViLT model's weights into vit_base_patch32_384 model given load_path is provided
        # no consideration of max_text_len != 40 case now
        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

    # ===================== Inference ===================== #
    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
        is_train=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"    # imgkey = "image_0"
        else:
            imgkey = "image"                                # imgkey = "image"

        # image embeddings - pass image data through ViT's visual_embed module to get image_embeds, etc.
        if image_embeds is None and image_masks is None:
            img = batch[imgkey][0]                          # shape: batch_size, 3, H, W
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,   # None since mask_it=False
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
            )
        else:
            patch_index, image_labels = (
                None,
                None,
            )
        
        # text embeddings - pass EHR data through EHREmbeddings module to get text_embeds
        text_embeds = self.text_embeddings(batch["text"])           # shape: batch_size, max_text_len-1, ehr_n_var -> batch_size, max_text_len, hidden_size     ### added

        # add modal-type embeddings (tensors of 0s for text, 1s for image) to text and image embeddings; shapes are unchanged
        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_embeds[:, :, 0]).long()),               ### modified
            image_embeds + self.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx))
        )

        ### added
        # prepend cls_token to masks of EHR data
        text_masks = batch["mask"]                                                    # shape: batch_size, max_text_len-1   ### added
        cls_token = torch.zeros((text_masks.size(0), 1), device=text_masks.device)
        text_masks = torch.cat([cls_token, text_masks], dim=1)                        # shape: batch_size, max_text_len

        # concatenate text and image embeddings, and text and image masks (by columns)
        # co_embeds = torch.cat([text_embeds, image_embeds], dim=1)       # shape: batch_size, max_text_len + {image_embeds.size(1)}, hidden_size
        # co_masks = torch.cat([text_masks, image_masks], dim=1)          # shape: batch_size, max_text_len + {image_masks.size(1)}

        ### pass only text embeddings and masks through transformer
        x = text_embeds
        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=text_masks)    # call each Block class instance's forward() method

        # normalise the concatenated transformer output
        x = self.transformer.norm(x)

        # separate text and image features from transformer output
        text_feats, image_feats = (x[:, : text_embeds.shape[1]], x[:, text_embeds.shape[1] :])

        # pass the cls_token from transformer output through pooling layer to get cls_feats
        cls_feats = self.pooler(x)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_labels_mppd": image_feats,
            "image_masks": image_masks,
            "text_masks": text_masks,
            "patch_index": patch_index,
        }
        return ret

    # ===================== Forward pass ===================== #
    def forward(self, batch):
        ret = dict()

        # if current_tasks is empty, return the output of infer()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret
        
        # call corresponding compute_ function from objectives.py for each task in current_tasks
        # this function calls infer() to get the embeddings and masks, and then passes them through the corresponding task-specific classifier
        # then, the output of the classifier is passed through the corresponding loss function to get the loss
        # finally, the loss is added to the ret dict

        # Masked Language Modeling objective
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Masked Patch Prediction objective
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))
            
        # Masked Patch Prediction objective
        if "mppd" in self.current_tasks:
            ret.update(objectives.compute_mppd(self, batch))

        # Image Text Matching objective
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_wpa(self, batch))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))
            
        # Binary classification for EHR+CXR dataset     ### added
        if "ehr_cxr" in self.current_tasks:
            ret.update(objectives.compute_ehr_cxr(self, batch))
        
        # Binary classification for Hateful Memes
        if "hatememes" in self.current_tasks:
            ret.update(objectives.compute_hatememes(self, batch))
            
        # Classification for Food101
        if "food101" in self.current_tasks:
            ret.update(objectives.compute_food101(self, batch))        
            
        # Multi-label classification for MM-IMDb
        if "mmimdb" in self.current_tasks:
            ret.update(objectives.compute_mmimdb(self, batch))
            
        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        # Image Retrieval and Text Retrieval for Flickr30k and MSCOCO
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)       # set current_tasks to the tasks in config["loss_names"] with non-zero values, e.g. ["mmimdb", "itm"]
        output = self(batch)            # call forward() with batch as input argument
        total_loss = sum([v for k, v in output.items() if "loss" in k])  # sum up all losses (from different tasks if there are multiple)
        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)   # compute and log epoch-level metrics, ie accuracy_epoch, AUROC_epoch, loss_epoch, the_metric

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)   # compute and log epoch-level metrics, ie accuracy_epoch, AUROC_epoch, loss_epoch, the_metric

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)            # call forward() with batch as input argument
        ret = dict()
        return ret                      # but return an empty dict since it's a test step

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)