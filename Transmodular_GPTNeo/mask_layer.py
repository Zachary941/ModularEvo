import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import copy
from functools import reduce
from tqdm import tqdm

class MaskLinear(nn.Linear):
    def __init__(self, *args, weight, bias, is_modular, is_binary=True, **kwargs):
        super(MaskLinear, self).__init__(*args, **kwargs)
        self.weight = weight  # load the weight and bias of pre-trained Linear.
        self.bias = bias

        self.is_modular = is_modular
        self.weight_mask, self.bias_mask = None, None
        if self.is_modular:
            self.init_mask()

        self.is_binary = is_binary

    def init_mask(self):
        self.weight_mask = nn.Parameter(torch.rand(self.weight.size()))
        if self.bias is not None:
            self.bias_mask = nn.Parameter(torch.rand(self.bias.size()))

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False


    def process_mask(self, mask):
        if self.is_binary:
            processed_mask = Binarization.apply(mask)
        else:
            processed_mask = torch.nn.functional.hardtanh(mask, min_val=0)
        return processed_mask

    def forward(self, inputs):
        if self.is_modular:
            weight_mask = self.process_mask(self.weight_mask)
            weight = self.weight * weight_mask
            if self.bias is not None:
                bias_mask = self.process_mask(self.bias_mask)
                bias = self.bias * bias_mask
            else:
                bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        output = F.linear(inputs, weight, bias)

        return output


class Binarization(autograd.Function):
    @staticmethod
    def forward(ctx, mask):
        bin_mask = (mask > 0).float()
        return bin_mask

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


def init_mask_model(model, no_mask: list = [], is_binary=True):
    mask_model = copy.deepcopy(model)

    for name, layer in model.named_modules():
        if any(nmn in name for nmn in no_mask):
            continue

        if isinstance(layer, torch.nn.Linear):
            obj_names = name.split('.')
            parent_obj = mask_model
            target_obj = None
            for obj_n in obj_names[:-1]:
                target_obj = getattr(parent_obj, obj_n)
                parent_obj = target_obj

            mask_layer = MaskLinear(in_features=layer.in_features, out_features=layer.out_features,
                                    weight=layer.weight, bias=layer.bias, is_modular=True, is_binary=is_binary)
            setattr(parent_obj, obj_names[-1], mask_layer)
        elif isinstance(layer, torch.nn.LayerNorm):
            # TODO
            pass
    return mask_model

    
def get_retained_word_ids(train_dataloader, downstream_task):
    assert downstream_task in ['Clone-detection', 'code-to-text']

    retained_word_ids = set()
    bar = tqdm(train_dataloader, total=len(train_dataloader), ncols=100, desc='filter embedding')

    if downstream_task == 'Clone-detection':
        for step, batch in enumerate(bar):
            inputs = batch[0].to('cuda')
            inputs = inputs.reshape(-1)
            inputs_set = set(inputs.cpu().tolist())
            retained_word_ids = retained_word_ids.union(inputs_set)
    elif downstream_task == 'code-to-text':
        for step, batch in enumerate(bar):
            batch = tuple(t.to('cuda') for t in batch)
            source_ids, source_mask, target_ids, target_mask = batch

            source_ids = source_ids.reshape(-1)
            source_set = set(source_ids.cpu().tolist())
            retained_word_ids = retained_word_ids.union(source_set)

            target_ids = target_ids.reshape(-1)
            target_set = set(target_ids.cpu().tolist())
            retained_word_ids = retained_word_ids.union(target_set)
    else:
        raise ValueError
    return retained_word_ids


def filter_embedding(encoder, train_dataloader, tokenizer, downstream_task):
    retained_word_ids = get_retained_word_ids(train_dataloader, downstream_task)

    retained_word_ids = retained_word_ids.union(set(tokenizer.all_special_ids))
    retained_word_ids = list(sorted(list(retained_word_ids)))

    # init vocab and embedding
    vocab = tokenizer.get_vocab()
    embedding = encoder.embeddings.word_embeddings.weight

    with torch.no_grad():
        new_embedding = torch.stack([v for i, v in enumerate(embedding) if i in retained_word_ids])

    vocab_to_retained = dict([(vocab_id, i) for i, vocab_id in enumerate(retained_word_ids)])

    # old ids to new ids
    ids_old2new = []
    for vocab_id in range(len(vocab)):
        if vocab_id in vocab_to_retained:
            ids_old2new.append((vocab_id, vocab_to_retained[vocab_id]))
        else:
            ids_old2new.append((vocab_id, 3))

    new_embedding_layer = torch.nn.Embedding.from_pretrained(new_embedding).to('cuda')
    embedding_layer = torch.nn.Embedding.from_pretrained(embedding).to('cuda')

    ids_vocab_old2new = []
    for emb_id, new_emb_id in ids_old2new:
        ids_vocab_old2new.append(torch.FloatTensor([new_emb_id]))
    ids_vocab_old2new = torch.stack(ids_vocab_old2new)
    ids_vocab_old2new_layer = torch.nn.Embedding.from_pretrained(ids_vocab_old2new).to('cuda')

    # new ids to old ids
    ids_vocab_new2old = []
    for emb_id, new_emb_id in ids_old2new:
        if emb_id != 3 and new_emb_id == 3:  # <unk>
            continue
        ids_vocab_new2old.append(torch.FloatTensor([emb_id]))
    assert len(ids_vocab_new2old) == len(new_embedding)
    ids_vocab_new2old = torch.stack(ids_vocab_new2old)
    ids_vocab_new2old_layer = torch.nn.Embedding.from_pretrained(ids_vocab_new2old).to('cuda')

    # check on training dataset, whether the original embedding layer and the filterd embedding layer generate the same embeddings given the same inputs.
    for data in train_dataloader:
        inputs = data[0].to('cuda')
        input_embs = embedding_layer(inputs)

        new_inputs = ids_vocab_old2new_layer(inputs)
        new_inputs = new_inputs.long()
        new_inputs = new_inputs.squeeze(-1)
        new_input_embs = new_embedding_layer(new_inputs)

        if not torch.equal(input_embs, new_input_embs):
            raise ValueError('not equal')

    print(f'\nNumber of embeddings: {encoder.embeddings.word_embeddings.num_embeddings} -> {len(new_embedding)}\n')

    # replace the original embedding layer with the filtered one.
    encoder.embeddings.word_embeddings.weight.data = new_embedding.data
    encoder.embeddings.word_embeddings.num_embeddings = len(new_embedding)
    return encoder, ids_vocab_old2new_layer, ids_vocab_new2old_layer


# different from the loss of wrr in modularizer.py, this function computes loss_wrr based on continuous masks.
def cal_weight_retention_rate(mask_model):
    masks = []
    for n, layer in mask_model.named_modules():
        if hasattr(layer, 'weight_mask'):
            masks.append(torch.flatten(layer.weight_mask))
            if layer.bias_mask is not None:
                masks.append(torch.flatten(layer.bias_mask))

    masks = torch.cat(masks, dim=0)
    masks = torch.nn.functional.hardtanh(masks, min_val=0)
    loss_wrr = torch.mean(masks)
    wrr = torch.mean((masks > 0).float())
    return loss_wrr, wrr