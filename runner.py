import os
import sys
import math
import glob
import uuid
import shutil
import random
import tempfile
import importlib
from pathlib import Path
from ipdb import set_trace
import torch
import torch.optim as optim
import torchaudio
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import is_initialized, get_rank, get_world_size
from learn2learn.algorithms.maml import MAML
from learn2learn.utils import clone_module
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from s3prl import hub
from s3prl.optimizers import get_optimizer
from s3prl.schedulers import get_scheduler
from s3prl.upstream.interfaces import Featurizer
from s3prl.utility.helper import is_leader_process, get_model_state, show, defaultdict

from huggingface_hub import HfApi, HfFolder, Repository

SAMPLE_RATE = 16000

MODEL_CARD_MARKDOWN = """---
datasets:
- superb
tags:
- library:s3prl
- benchmark:superb
- type:model
---

# Fine-tuned s3prl model

Upstream Model: {upstream_model}

## Model description

[More information needed]

## Intended uses & limitations

[More information needed]

## How to use

[More information needed]

## Limitations and bias

[More information needed]

## Training data

[More information needed]

## Training procedure

[More information needed]

## Evaluation results

[More information needed]

"""

class CustomModel(torch.nn.Module):
    def __init__(self, components):
        super(CustomModel, self).__init__()
        assert len(components) == 4, "components list must have exactly three elements: [upstream, featurizer, downstreamProjector, downstreamModel]"
        self.upstream = components[0]
        self.featurizer = components[1]
        self.downstreamProjector = components[2]
        self.downstreamModel = components[3]

    def forward(self, wavs):
        # Forward pass through the upstream model
        features = self.upstream(wavs)

        # Forward pass through the featurizer model
        features = self.featurizer(wavs, features)

        # Forward pass through the downstream model
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=device)

        features = pad_sequence(features, batch_first=True)
        features = self.downstreamProjector(features)
        predicted, _ = self.downstreamModel(features, features_len)

        return predicted

class ModelEntry:
    def __init__(self, model, name, trainable, interfaces):
        self.model = model
        self.name = name
        self.trainable = trainable
        self.interfaces = interfaces


class Runner():
    """
    Used to handle high-level concepts of a ML experiment
    eg. training loop, evaluation loop, upstream propagation, optimization, logging, checkpoint saving
    """
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.init_ckpt = torch.load(self.args.init_ckpt, map_location='cpu') if self.args.init_ckpt else {}

        self.upstream = self._get_upstream()
        self.featurizer = self._get_featurizer()
        self.downstream = self._get_downstream()
        self.all_entries = [self.upstream, self.featurizer, self.downstream]

    def _load_weight(self, model, name):

        init_weight = self.init_ckpt.get(name)
        if init_weight:
            show(f'[Runner] - Loading {name} weights from the previous experiment')
            model.load_state_dict(init_weight)


    def _init_model(self, model, name, trainable, interfaces=None):
        for interface in interfaces or []:
            assert hasattr(model, interface), interface
        # set_trace()
        self._load_weight(model, name)

        if is_initialized() and trainable and any((p.requires_grad for p in model.parameters())):

            model = DDP(model, device_ids=[self.args.local_rank], find_unused_parameters=True)
            
            for interface in interfaces or []:
                setattr(model, interface, getattr(model.module, interface))

        return ModelEntry(model, name, trainable, interfaces)


    def _get_upstream(self):
        if "from_hf_hub" in self.args and self.args.from_hf_hub == True:
            from huggingface_hub import snapshot_download

            print(f'[Runner] - Downloading upstream model {self.args.upstream} from the Hugging Face Hub')
            filepath = snapshot_download(self.args.upstream, self.args.upstream_revision, use_auth_token=True)
            sys.path.append(filepath)

            dependencies = (Path(filepath) / 'requirements.txt').resolve()
            print("[Dependency] - The downloaded upstream model requires the following dependencies. Please make sure they are installed:")
            for idx, line in enumerate((Path(filepath) / "requirements.txt").open().readlines()):
                print(f"{idx}. {line.strip()}")
            print(f"You can install them by:")
            print()
            print(f"pip install -r {dependencies}")
            print()

            from expert import UpstreamExpert
            Upstream = UpstreamExpert
            ckpt_path = os.path.join(filepath, self.args.upstream_model_name)
        else:
            Upstream = getattr(hub, self.args.upstream)
            ckpt_path = self.args.upstream_ckpt
        upstream_refresh = self.args.upstream_refresh

        if is_initialized() and get_rank() > 0:
            torch.distributed.barrier()
            upstream_refresh = False
        #set_trace()
        model = Upstream(
            ckpt = ckpt_path,
            model_config = self.args.upstream_model_config,
            refresh = upstream_refresh,
        ).to(self.args.device)

        if is_initialized() and get_rank() == 0:
            torch.distributed.barrier()

        return self._init_model(
            model = model,
            name = 'Upstream',
            trainable = self.args.upstream_trainable,
            interfaces = ["get_downsample_rates"]
        )

    def _get_featurizer(self):
        model = Featurizer(
            upstream = self.upstream.model,
            feature_selection = self.args.upstream_feature_selection,
            layer_selection = self.args.upstream_layer_selection,
            upstream_device = self.args.device,
            normalize = self.args.upstream_feature_normalize,
        ).to(self.args.device)

        return self._init_model(
            model = model,
            name = 'Featurizer',
            trainable = True,
            interfaces = ['output_dim', 'downsample_rate']
        )


    def _get_downstream(self):
        expert = importlib.import_module(f"s3prl.downstream.{self.args.downstream}.expert")
        Downstream = getattr(expert, "DownstreamExpert")
        
        model = Downstream(
            upstream_dim = self.featurizer.model.output_dim,
            upstream_rate = self.featurizer.model.downsample_rate,
            **self.config,
            **vars(self.args)
        ).to(self.args.device)

        return self._init_model(
            model = model,
            name = 'Downstream',
            trainable = True,
            interfaces = ['get_dataloader', 'log_records']
        )


    def _get_optimizer(self, model_params):
        optimizer = get_optimizer(
            model_params, 
            self.config['runner']['total_steps'],
            self.config['optimizer']
        )
        self._load_weight(optimizer, 'Optimizer')
        return optimizer

    def _get_scheduler(self, optimizer):
        scheduler = get_scheduler(
            optimizer,
            self.config['runner']['total_steps'],
            self.config['scheduler']
        )
        self._load_weight(scheduler, 'Scheduler')
        return scheduler

    def _create_model_card(self, path):
        model_card = MODEL_CARD_MARKDOWN.format(upstream_model=self.args.upstream)
        with open(os.path.join(path, "README.md"), "w") as f:
            f.write(model_card)
    
    def split_data(self, wavs, labels):
        num_samples = len(wavs)

        # 70% adaptation, 30% evaluation
        #should be modified as needed
        adaptation_proportion = 0.7
        num_adaptation_samples = int(adaptation_proportion * num_samples)

        # Randomly shuffle the indices
        indices = torch.randperm(num_samples).to(self.args.device)

        # Split indices into adaptation and evaluation
        adaptation_indices = indices[:num_adaptation_samples]
        evaluation_indices = indices[num_adaptation_samples:]

        # Use the indices to split the data
        adaptation_wavs = [wavs[i] for i in adaptation_indices]
        adaptation_labels = labels[adaptation_indices]

        evaluation_wavs = [wavs[i] for i in evaluation_indices]
        evaluation_labels = labels[evaluation_indices]

        return adaptation_wavs, adaptation_labels, evaluation_wavs, evaluation_labels

    def evaluate_model(self, clone, evaluation_wavs):
        predictions = clone(evaluation_wavs)
        normalized_preds = torch.nn.functional.softmax(predictions, dim=1).to(self.args.device)
        target_preds = (1.0 / predictions.shape[1]) * torch.ones((predictions.shape[0], predictions.shape[1])).to(self.args.device)
        evaluation_error = F.kl_div(torch.log(normalized_preds), target_preds, reduction='batchmean')
    
        return evaluation_error

    def evaluate_model_accuracy(self, clone, test_dataloader_name="test"):
        # Get the test dataloader
        test_dataloader = self.downstream.model.get_dataloader(self.config['runner'].get("test_dataloader", test_dataloader_name))
        tqdm_file = sys.stderr if is_leader_process() else open(os.devnull, 'w')
        # Set the model to evaluation mode
        clone.eval()
        
        # Initialize accuracy counter and total samples counter
        acc = 0
        total_samples = 0
        
        # Iterate through the test dataloader
        for batch_id, (wavs, labels, *others) in enumerate(tqdm(test_dataloader, dynamic_ncols=True, desc='test', file=tqdm_file)):
            # Move wavs and labels to the correct device
            wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]
            labels = torch.LongTensor(labels).to(self.args.device)
            
            # Make predictions
            predicted = clone(wavs)
            predicted_classid = predicted.argmax(dim=-1)  # Get the class with the highest score
            
            # Calculate the number of correct predictions
            correct_predictions = (predicted_classid == labels).sum().item()
            
            # Update accuracy counter and total samples
            acc += correct_predictions
            total_samples += labels.size(0)
        
        # Calculate the final accuracy
        accuracy = acc / total_samples
        
        return accuracy

    def benignTrain(self, upstream = None):
        criterion = torch.nn.CrossEntropyLoss()
        if upstream == None:
            upstream = self.upstream.model
        else:
            self.downstream = self._get_downstream()
        train_dataloader = self.downstream.model.get_dataloader(self.config['runner'].get("train_dataloader", "train"))
        tqdm_file = sys.stderr if is_leader_process() else open(os.devnull, 'w')
        model = CustomModel([upstream, self.featurizer.model, self.downstream.model.projector, self.downstream.model.model]).to(self.args.device)
       #optimizer1 = optim.Adam(upstream.parameters(), lr=0.001)
        optimizer2 = optim.Adam(self.downstream.model.parameters(), lr=0.001)

        print('Acc before= ', self.evaluate_model_accuracy(model))
        train_split = self.config['runner'].get("train_dataloader", "train")
        records = defaultdict(list)
        for k in range(8):
            tLoss = 0
            for batch_id, (wavs, labels, *others) in enumerate(tqdm(train_dataloader, dynamic_ncols=True, desc='train', file=tqdm_file)):
                wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]
                labels = torch.LongTensor(labels).to(self.args.device)
                #adaptation_wavs, adaptation_labels, evaluation_wavs, evaluation_labels = self.split_data(wavs, labels)
                optimizer2.zero_grad()
                predictions = model(wavs)
                loss = criterion(predictions, labels)
                loss.backward()
                tLoss += loss.item()
                optimizer2.step()
            
            print(f'tLoss: {tLoss / len(train_dataloader)}')
            print(self.evaluate_model_accuracy(model))

    def sophon(self):
        pretrain_dataloader = self.downstream.model.load_dataset('train')
        #pretrain_test_dataloader = self.downstream.model.load_dataset('valid')
        model = CustomModel([self.upstream.model, self.featurizer.model, self.downstream.model.projector, self.downstream.model.model]).to(self.args.device)
        criterion = torch.nn.CrossEntropyLoss()
        train_dataloader = self.downstream.model.get_dataloader(self.config['runner'].get("train_dataloader", "train"))
        tqdm_file = sys.stderr if is_leader_process() else open(os.devnull, 'w')
        optimizer1 = optim.Adam(self.upstream.model.parameters(), lr=0.0001)
        optimizer2 = optim.Adam(self.downstream.model.parameters(), lr=0.0001)
        lossSu = 0
        print('Acc before= ', self.evaluate_model_accuracy(model))
        for k in range(10):
            tLoss = 0
            eLoss = 0
            optimizer1.zero_grad()
            for batch_id, (wavs, labels, *others) in enumerate(tqdm(train_dataloader, dynamic_ncols=True, desc='train', file=tqdm_file)):
                wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]
                labels = torch.LongTensor(labels).to(self.args.device)
                adaptation_wavs, adaptation_labels, evaluation_wavs, evaluation_labels = self.split_data(wavs, labels)
                optimizer2.zero_grad()
                predictions = model(adaptation_wavs)
                loss = criterion(predictions, adaptation_labels)

                #dont update upstream model grad when doing adaption loss
                for param in model.upstream.parameters():
                    param.requires_grad = False
                
                loss.backward()
                tLoss += loss.item()
                optimizer2.step()

                for param in model.upstream.parameters():
                    param.requires_grad = True

                optimizer2.zero_grad()
                evaluation_error = self.evaluate_model(model, evaluation_wavs)
                eLoss += evaluation_error.item()
                evaluation_error.backward()
                optimizer2.zero_grad()
            
            print(f'tLoss: {tLoss / len(train_dataloader)}')
            print(f'eLoss: {eLoss / len(train_dataloader)}')
            #print("before")
            #print(self.evaluate_model_accuracy(model))
            optimizer1.step()
            #print("after")
            torch.cuda.empty_cache()
            print(self.evaluate_model_accuracy(model))

        torch.cuda.empty_cache()
        print("GOT HERE")
        
        self.downstream.model.train()
        iterIdx = 0
        nilIters = 40
        for batch_id, data in enumerate(pretrain_dataloader):
            if iterIdx > nilIters:
                break
            iterIdx +=1
            optimizer1.zero_grad()
            loss = 0.0
            reduction = "sum"
            tmp = [d.to(self.args.device) for d in data["target_list"]]
            net_input = {key: value.to(self.args.device) for key, value in data["net_input"].items()}
            self.upstream.model.model.to(self.args.device)
            net_output = self.upstream.model.model(target_list=tmp,  source = net_input['source'].to(self.args.device), padding_mask= net_input['padding_mask'].to(self.args.device))
            loss_m_list = []
            logp_m_list = self.upstream.model.model.get_logits(net_output, True)
            targ_m_list = self.upstream.model.model.get_targets(net_output, True)
            for i, (logp_m, targ_m) in enumerate(zip(logp_m_list, targ_m_list)):
                loss_m = F.cross_entropy(logp_m, targ_m, reduction=reduction)
                lossSu += loss_m.item()
                loss_m.backward()
            optimizer1.step()

            torch.cuda.empty_cache()
            del data
        
        print(f'PreTrainingLoss: {lossSu / nilIters }')

        self.benignTrain(self.upstream.model)

    def train(self):
        # trainable parameters and train/eval mode
        trainable_models = []
        trainable_paras = []
        for entry in self.all_entries:
            if entry.trainable:
                entry.model.train()
                trainable_models.append(entry.model)
                trainable_paras += list(entry.model.parameters())
            else:
                entry.model.eval()

        # optimizer
        optimizer = self._get_optimizer(trainable_models)

        # scheduler
        scheduler = None
        if self.config.get('scheduler'):
            scheduler = self._get_scheduler(optimizer)

        # specaug
        specaug = None
        if self.config.get('specaug'):
            from .specaug import SpecAug
            specaug = SpecAug(**self.config["specaug"])

        # progress bar
        tqdm_file = sys.stderr if is_leader_process() else open(os.devnull, 'w')
        pbar = tqdm(total=self.config['runner']['total_steps'], dynamic_ncols=True, desc='overall', file=tqdm_file)
        init_step = self.init_ckpt.get('Step')
        if init_step:
            pbar.n = init_step

        # Tensorboard logging
        if is_leader_process():
            logger = SummaryWriter(self.args.expdir)

        batch_ids = []
        backward_steps = 0
        records = defaultdict(list)
        epoch = self.init_ckpt.get('Epoch', 0)
        train_split = self.config['runner'].get("train_dataloader", "train")
        while pbar.n < pbar.total:
            try:
                dataloader = self.downstream.model.get_dataloader(train_split, epoch=epoch)
            except TypeError as e:
                if "unexpected keyword argument 'epoch'" in str(e):
                    dataloader = self.downstream.model.get_dataloader(train_split)
                    if hasattr(dataloader, "sampler") and isinstance(dataloader.sampler, DistributedSampler):
                        dataloader.sampler.set_epoch(epoch)
                else:
                    raise

            for batch_id, (wavs, *others) in enumerate(tqdm(dataloader, dynamic_ncols=True, desc='train', file=tqdm_file)):
                # try/except block for forward/backward
                try:
                    if pbar.n >= pbar.total:
                        break
                    global_step = pbar.n + 1
                    
                    wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]
                    if self.upstream.trainable:
                        features = self.upstream.model(wavs)
                    else:
                        with torch.no_grad():
                            features = self.upstream.model(wavs)
                    features = self.featurizer.model(wavs, features)

                    if specaug:
                        features, _ = specaug(features)

                    loss = self.downstream.model(
                        train_split,
                        features, *others,
                        records = records,
                    )
                    batch_ids.append(batch_id)

                    gradient_accumulate_steps = self.config['runner'].get('gradient_accumulate_steps')
                    (loss / gradient_accumulate_steps).backward()
                    del loss

                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print(f'[Runner] - CUDA out of memory at step {global_step}')
                        if is_initialized():
                            raise
                        with torch.cuda.device(self.args.device):
                            torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        continue
                    else:
                        raise

                # whether to accumulate gradient
                backward_steps += 1
                if backward_steps % gradient_accumulate_steps > 0:
                    continue

                # gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_paras, self.config['runner']['gradient_clipping'])

                # optimize
                if math.isnan(grad_norm):
                    print(f'[Runner] - grad norm is NaN at step {global_step}')
                else:
                    optimizer.step()
                optimizer.zero_grad()

                # adjust learning rate
                if scheduler:
                    scheduler.step()

                if not is_leader_process():
                    batch_ids = []
                    records = defaultdict(list)
                    continue

                # logging
                if global_step % self.config['runner']['log_step'] == 0:
                    self.downstream.model.log_records(
                        train_split,
                        records = records,
                        logger = logger,
                        global_step = global_step,
                        batch_ids = batch_ids,
                        total_batch_num = len(dataloader),
                    )
                    batch_ids = []
                    records = defaultdict(list)

                # evaluation and save checkpoint
                save_names = []

                if global_step % self.config['runner']['eval_step'] == 0:
                    for split in self.config['runner']['eval_dataloaders']:
                        save_names += self.evaluate(split, logger, global_step)

                if global_step % self.config['runner']['save_step'] == 0:
                    def check_ckpt_num(directory):
                        max_keep = self.config['runner']['max_keep']
                        ckpt_pths = glob.glob(f'{directory}/states-*.ckpt')
                        if len(ckpt_pths) >= max_keep:
                            ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
                            for ckpt_pth in ckpt_pths[:len(ckpt_pths) - max_keep + 1]:
                                os.remove(ckpt_pth)
                    check_ckpt_num(self.args.expdir)
                    save_names.append(f'states-{global_step}.ckpt')

                if len(save_names) > 0:
                    all_states = {
                        'Optimizer': optimizer.state_dict(),
                        'Step': global_step,
                        'Epoch': epoch,
                        'Args': self.args,
                        'Config': self.config,
                    }

                    for entry in self.all_entries:
                        if entry.trainable:
                            all_states[entry.name] = get_model_state(entry.model)

                    if scheduler:
                        all_states['Scheduler'] = scheduler.state_dict()

                    if is_initialized():
                        all_states['WorldSize'] = get_world_size()

                    save_paths = [os.path.join(self.args.expdir, name) for name in save_names]
                    tqdm.write(f'[Runner] - Save the checkpoint to:')
                    for i, path in enumerate(save_paths):
                        tqdm.write(f'{i + 1}. {path}')
                        torch.save(all_states, path)

                pbar.update(1)
            epoch += 1

        pbar.close()

        if self.args.push_to_hf_hub:
            self.push_to_huggingface_hub()
        if is_leader_process():
            logger.close()


    def evaluate(self, split=None, logger=None, global_step=0):

        """evaluate function will always be called on a single process even during distributed training"""

        # When this member function is called directly by command line
        not_during_training = split is None and logger is None and global_step == 0
        if not_during_training:
            split = self.args.evaluate_split
            tempdir = tempfile.mkdtemp()
            logger = SummaryWriter(tempdir)

        # fix seed to guarantee the same evaluation protocol across steps 
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
            with torch.cuda.device(self.args.device):
                torch.cuda.empty_cache()

        # record original train/eval states and set all models to eval
        trainings = []
        for entry in self.all_entries:
            trainings.append(entry.model.training)
            entry.model.eval()

        # prepare data
        dataloader = self.downstream.model.get_dataloader(split)
        evaluate_ratio = float(self.config["runner"].get("evaluate_ratio", 1))
        evaluate_steps = round(len(dataloader) * evaluate_ratio)

        batch_ids = []
        records = defaultdict(list)

        for batch_id, (wavs, *others) in enumerate(tqdm(dataloader, dynamic_ncols=True, desc=split, total=evaluate_steps)):
            #print(others)
            if batch_id > evaluate_steps:
                break

            wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]

            with torch.no_grad():
                features = self.upstream.model(wavs)
                features = self.featurizer.model(wavs, features)
                self.downstream.model(
                    split,
                    features, *others,
                    records = records,
                    batch_id = batch_id,
                )
                batch_ids.append(batch_id)

        save_names = self.downstream.model.log_records(
            split,
            records = records,
            logger = logger,
            global_step = global_step,
            batch_ids = batch_ids,
            total_batch_num = len(dataloader),
        )
        batch_ids = []
        records = defaultdict(list)

        # prepare back to training
        if torch.cuda.is_available():
            with torch.cuda.device(self.args.device):
                torch.cuda.empty_cache()

        for entry, training in zip(self.all_entries, trainings):
            if training:
                #entry.model.train()
                entry.model.train().to(self.args.device)#

        if not_during_training:
            logger.close()
            shutil.rmtree(tempdir)

        return [] if type(save_names) is not list else save_names

    def inference(self):
        filepath = Path(self.args.evaluate_split)
        assert filepath.is_file(), filepath
        filename = filepath.stem

        if hasattr(self.downstream.model, "load_audio"):
            wav = self.downstream.model.load_audio(filepath)
        else:
            wav, sr = torchaudio.load(str(filepath))
            assert sr == SAMPLE_RATE, sr
        wavs = [wav.view(-1).to(self.args.device)]

        for entry in self.all_entries:
            entry.model.eval()

        with torch.no_grad():
            features = self.upstream.model(wavs)
            features = self.featurizer.model(wavs, features)
            self.downstream.model.inference(features, [filename])

    def push_to_huggingface_hub(self):
        """Creates a downstream repository on the Hub and pushes training artifacts to it."""
        if self.args.hf_hub_org.lower() != "none":
            organization = self.args.hf_hub_org
        else:
            organization = os.environ.get("HF_USERNAME")
        huggingface_token = HfFolder.get_token()
        print(f"[Runner] - Organisation to push fine-tuned model to: {organization}")
        
        # Extract upstream repository metadata
        if self.args.hub == "huggingface":
            model_info = HfApi().model_info(self.args.upstream, token=huggingface_token)
            downstream_model_id = model_info.sha
            # Exclude "/" characters from downstream repo ID
            upstream_model_id = model_info.modelId.replace("/", "__")
        else:
            upstream_model_id = self.args.upstream.replace("/", "__")
            downstream_model_id = str(uuid.uuid4())[:8]
        repo_name = f"{upstream_model_id}__{downstream_model_id}"
        # Create downstream repo on the Hub
        repo_url = HfApi().create_repo(
            token=huggingface_token,
            name=repo_name,
            organization=organization,
            exist_ok=True,
            private=False,
        )
        print(f"[Runner] - Created Hub repo: {repo_url}")

        # Download repo
        HF_HUB_DIR = "hf_hub"
        REPO_ROOT_DIR = os.path.join(self.args.expdir, HF_HUB_DIR, repo_name)
        REPO_TASK_DIR = os.path.join(REPO_ROOT_DIR, self.args.downstream, self.args.expname)
        print(f"[Runner] - Cloning Hub repo to {REPO_ROOT_DIR}")
        model_repo = Repository(
            local_dir=REPO_ROOT_DIR, clone_from=repo_url, use_auth_token=huggingface_token
        )
        # Pull latest changes if they exist
        model_repo.git_pull()

        # Copy checkpoints, tensorboard logs, and args / configs
        # Note that this copies all files from the experiment directory,
        # including those from multiple runs
        shutil.copytree(self.args.expdir, REPO_TASK_DIR, dirs_exist_ok=True, ignore=shutil.ignore_patterns(HF_HUB_DIR))

        # By default we use model.ckpt in the PreTrainedModel interface, so
        # rename the best checkpoint to match this convention
        checkpoints = list(Path(REPO_TASK_DIR).glob("*best*.ckpt"))
        if len(checkpoints) == 0:
            print("[Runner] - Did not find a best checkpoint! Using the final checkpoint instead ...")
            CKPT_PATH = (
                os.path.join(REPO_TASK_DIR, f"states-{self.config['runner']['total_steps']}.ckpt")
                )
        elif len(checkpoints) > 1:
            print(f"[Runner] - More than one best checkpoint found! Using {checkpoints[0]} as default ...")
            CKPT_PATH = checkpoints[0]
        else:
            print(f"[Runner] - Found best checkpoint {checkpoints[0]}!")
            CKPT_PATH = checkpoints[0]
        shutil.move(CKPT_PATH, os.path.join(REPO_TASK_DIR, "model.ckpt"))
        model_repo.lfs_track("*.ckpt")

        # Write model card
        self._create_model_card(REPO_ROOT_DIR)

        # Push everything to the Hub
        print("[Runner] - Pushing model files to the Hub ...")
        model_repo.push_to_hub()
        print("[Runner] - Training run complete!")
