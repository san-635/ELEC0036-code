from sacred import Experiment
ex = Experiment("ViLT")


def _loss_names(d):
    ret = {
        "itm": 0,           # image-text matching
        "mlm": 0,           # masked language modelling
        "mpp": 0,           # masked patch prediction
        "mppd": 0,          # masked patch prediction

        "vqa": 0,           # visual question answering (VQAv2)
        "nlvr2": 0,         # natural language for visual reasoning 2 (NLVR2)
        "irtr": 0,          # image retrieval text retrieval (Flickr30K, MSCOCO)
        "mmimdb": 0,        # MM-IMDb
        "hatememes": 0,     # Hateful Memes
        "food101": 0,       # UPMC Food101

        "ehr_cxr": 0,       # EHR+CXR dataset   ### added
        "ehr_ppg_cxr": 0,   # EHR+PPG+CXR dataset   ### added
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "vilt"
    seed = 0    # for reproducibility of weights initialization and dataloader shuffling
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # eval config (for bash execution)
    test_ratio = None
    test_type = None
    test_exp_name = None
    
    # fix backbone model (ViLT) weights
    fix_model = True
    
    # missing modality config
    missing_ratio = {'train': 0, 'val': 0, 'test': 0}   ### modified from 0.7 to 0 to avoid using missing prompts
    missing_type = {'train': 'both', 'val': 'both', 'test': 'both'} # ['text', 'image', 'both'] in VL tasks
    both_ratio = 0   ### modified from 0.5 to 0 to avoid using missing prompts
    missing_table_root = '' ### modified from './datasets/missing_tables/' to '' to avoid using missing prompts
    simulate_missing = False
    
    # missing_aware_prompts config
    prompt_type = 'input'
    prompt_length = 16
    learnt_p = True
    prompt_layers = [0,1,2,3,4,5]
    multi_layer_prompt = True    
        
    # Image setting
    train_transform_keys = ["pixelbert"]
    val_transform_keys = ["pixelbert"]
    image_size = 384
    max_image_len = -1
    patch_size = 32
    draw_false_image = 1
    image_only = False

    # Text Setting
    vqav2_label_size = 3129
    max_text_len = 40
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0
    ehr_n_var = 16              ### added
    timestep = 0                ### added
    impute_strategy = "zero"    ### added

    # PPG Setting               ### added
    max_ppg_len = 0
    ppg_overlap = 0
    k_folds = 0

    # Transformer Setting
    vit = "vit_base_patch32_384"
    hidden_size = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-4
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = 25000
    warmup_steps = 2500
    end_lr = 0
    lr_mult = 1  # multiply by lr for downstream classifier heads

    # Downstream Setting
    get_recall_metric = False
    class_num = 2    ### added
    mmimdb_class_num = 23
    hatememes_class_num = 2
    food101_class_num = 101

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False
    finetune_first = False

    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 0   # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    gpu_device = 3          # the DEV # of the GPU to use
    load_path = ""          # path to pre-trained ViLT model's weights
    num_workers = 8         # number of subprocesses for torch.utils.data.DataLoader    ### modified from 8 to 4 to avoid memory error
    precision = 16

### added
@ex.named_config
def task_finetune_ehr_cxr():
    data_root = "./ehr_cxr_dataset"     # ./ehr_cxr_dataset_comp when comparing with EHR+PPG+CXR dataset
    exp_name = "finetune_ehr_cxr"
    loss_names = _loss_names({"ehr_cxr": 1})
    batch_size = 256            # 6 for _44 (EHR and EHR+CXR)
    per_gpu_batchsize = 256     # 6 for _44 (EHR and EHR+CXR)
    max_epoch = 20              # 30 for _44 (EHR and EHR+CXR)
    max_steps = None
    warmup_steps = 0.1          # 0 for EHR+CXR_44 and 0.1 for EHR_44
    decay_power = 1             # same for _44 (EHR and EHR+CXR)
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.5
    weight_decay = 2e-2
    max_text_len = 97           # 48h / 0.5h + 1; sample every 0.5h for 48h
    timestep = 0.5
    impute_strategy = "zero"

### added
@ex.named_config
def task_finetune_ehr_ppg_cxr():
    data_root = "./ehr_ppg_cxr_dataset"
    exp_name = "finetune_ehr_ppg_cxr"
    loss_names = _loss_names({"ehr_ppg_cxr": 1})
    batch_size = 6
    per_gpu_batchsize = 6
    max_epoch = 25
    max_steps = None
    warmup_steps = 0.1
    decay_power = "cosine"
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.5
    weight_decay = 2e-2
    max_text_len = 97           # 48h / 0.5h + 1; sample every 0.5h for 48h
    timestep = 0.5
    impute_strategy = "zero"
    max_ppg_len = 51            # PPG window size + 1
    # ppg_overlap = 25            # overlap for PPG windows
    # k_folds = 5                 # 5-fold cross-validation; for _win_cv and _og_cv

# Named configs for "environment" which define gpus and nodes, and paths
@ex.named_config
def env_dandelin():
    data_root = "/data2/dsets/dataset"
    log_dir = "/data2/vilt/result"
    num_gpus = 8
    num_nodes = 1


# Named configs for "task" which define datasets, loss_names and desired batch_size, warmup_steps, epochs, and exp_name
@ex.named_config
def task_mlm_itm():
    exp_name = "mlm_itm"
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096
    max_epoch = 10
    max_image_len = 200


@ex.named_config
def task_mlm_itm_randaug():
    exp_name = "mlm_itm_randaug"
    datasets = ["coco", "vg", "sbu", "gcc"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096
    max_epoch = 10
    max_image_len = 200


@ex.named_config
def task_mlm_itm_mpp():
    exp_name = "mlm_itm_mpp"
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1, "mpp": 1})
    batch_size = 4096
    max_epoch = 10
    max_image_len = 200


@ex.named_config
def task_finetune_nlvr2():
    exp_name = "finetune_nlvr2"
    datasets = ["nlvr2"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4


@ex.named_config
def task_finetune_nlvr2_randaug():
    exp_name = "finetune_nlvr2_randaug"
    datasets = ["nlvr2"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4


@ex.named_config
def task_finetune_vqa():
    exp_name = "finetune_vqa"
    datasets = ["vqa"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10


@ex.named_config
def task_finetune_vqa_randaug():
    exp_name = "finetune_vqa_randaug"
    datasets = ["vqa"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10

@ex.named_config
def task_finetune_hatememes():
    data_root = "./datasets/Hatefull_Memes"
    exp_name = "finetune_hatememes"
    datasets = ["Hatefull_Memes"]
    loss_names = _loss_names({"hatememes": 1})
    batch_size = 256
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4        ### modified from 1e-2
    val_check_interval = 0.5    ### modified from 0.11
    weight_decay = 2e-2
#     optim_type = "adam"
    max_text_len = 128 
    
@ex.named_config
def task_finetune_food101():
    exp_name = "finetune_food101"
    datasets = ["Food101"]
    loss_names = _loss_names({"food101": 1})
    batch_size = 256
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-2
    val_check_interval = 0.2
    weight_decay = 2e-2
#     optim_type = "adam"
    max_text_len = 512     
    
@ex.named_config
def task_finetune_mmimdb():
    exp_name = "finetune_mmimdb"
    datasets = ["mmimdb"]
    loss_names = _loss_names({"mmimdb": 1})
#     loss_names = _loss_names({"mmimdb": 1, "prompt": -0.5})
    batch_size = 256
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-2
    val_check_interval = 0.2
    weight_decay = 2e-2
#     optim_type = "adam"
    max_text_len = 1024

@ex.named_config
def task_finetune_irtr_coco():
    exp_name = "finetune_irtr_coco"
    datasets = ["coco"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4


@ex.named_config
def task_finetune_irtr_coco_randaug():
    exp_name = "finetune_irtr_coco_randaug"
    datasets = ["coco"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4


@ex.named_config
def task_finetune_irtr_f30k():
    exp_name = "finetune_irtr_f30k"
    datasets = ["f30k"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4


@ex.named_config
def task_finetune_irtr_f30k_randaug():
    exp_name = "finetune_irtr_f30k_randaug"
    datasets = ["f30k"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4


# Named configs for "etc" which are orthogonal to "env" and "task", need to be added at the end
@ex.named_config
def step25k():
    max_epoch = 100
    max_steps = 25000


@ex.named_config
def step50k():
    max_epoch = 100
    max_steps = 50000


@ex.named_config
def step100k():
    max_epoch = 100
    max_steps = 100000


@ex.named_config
def step200k():
    max_epoch = 200
    max_steps = 200000


@ex.named_config
def vit32_base():
    vit = "vit_base_patch32_384"
    patch_size = 32
    hidden_size = 768
    num_heads = 12
    num_layers = 12
