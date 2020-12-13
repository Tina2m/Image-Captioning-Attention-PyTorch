# %%
from IPython.core.display import display
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

from utils_torch import *
from datasets.flickr8k import Flickr8kDataset

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device
# %%

DATASET_BASE_PATH = 'data/flickr8k/'

# %%

train_set = Flickr8kDataset(dataset_base_path=DATASET_BASE_PATH, dist='train', device=device)
vocab, word2idx, idx2word, max_len = vocab_set = train_set.get_vocab()
val_set = Flickr8kDataset(dataset_base_path=DATASET_BASE_PATH, dist='val', vocab_set=vocab_set, device=device)
test_set = Flickr8kDataset(dataset_base_path=DATASET_BASE_PATH, dist='test', vocab_set=vocab_set, device=device)
len(train_set), len(val_set), len(test_set)

# %%
vocab_size = len(vocab)
vocab_size, max_len

# %%

samples_per_epoch = len(train_set)
samples_per_epoch


# %%
def train_model(model, train_generator, steps_per_epoch, optimizer, loss_fn, wandb_log=False):
    running_acc = 0
    running_loss = 0.0

    t = trange(steps_per_epoch, leave=True)
    for batch_idx in t:  # enumerate(iter(steps_per_epoch)):
        batch = next(train_generator)
        (enc, cap_in, next_word) = batch

        optimizer.zero_grad()
        output = model(enc, cap_in)
        loss = loss_fn(output, next_word)
        loss.backward()
        optimizer.step()

        running_acc += (torch.argmax(output, dim=1) == next_word).sum().item() / next_word.size(0)
        running_loss += loss.item()
        t.set_postfix({'loss': running_loss / (batch_idx + 1),
                       'acc': running_acc / (batch_idx + 1)}, refresh=True)

    return model, running_loss


def train_model_new(train_loader, encoder, decoder, loss_fn, optimizer, vocab_size):
    running_acc = 0.0
    running_loss = 0.0
    encoder.train()
    decoder.train()
    t = tqdm(iter(train_loader))
    for batch_idx, batch in enumerate(t):  # enumerate(iter(steps_per_epoch)):
        images, captions = batch

        optimizer.zero_grad()
        features = encoder(images)
        outputs = decoder(features, captions)

        loss = loss_fn(outputs.view(-1, vocab_size), captions.view(-1))
        loss.backward()
        optimizer.step()

        running_acc += (torch.argmax(outputs.view(-1, vocab_size), dim=1) == captions.view(
            -1)).sum().item() / captions.view(-1).size(0)
        running_loss += loss.item()
        t.set_postfix({'loss': running_loss / (batch_idx + 1),
                       'acc': running_acc / (batch_idx + 1),
                       }, refresh=True)

    return running_loss / len(train_loader)


# %%

MODEL = "resnet50_monolstm"
EMBEDDING_DIM = 50
EMBEDDING = f"GLV{EMBEDDING_DIM}"
BATCH_SIZE = 16
LR = 1e-2
MODEL_NAME = f'saved_models/{MODEL}_b{BATCH_SIZE}_emd{EMBEDDING}'
NUM_EPOCHS = 2

# %%

from models.torch.resnet50_monolstm import Encoder

encoder = Encoder(embed_size=300).to(device=device)

# %%

from models.torch.resnet50_monolstm import Decoder

encoder = Encoder(embed_size=EMBEDDING_DIM).to(device)
decoder = Decoder(EMBEDDING_DIM, 256, vocab_size, num_layers=2).to(device)

# Define the loss function
loss_fn = torch.nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else torch.nn.CrossEntropyLoss()

# Specify the learnable parameters of the model
params = list(decoder.parameters()) + list(encoder.embed.parameters()) + list(encoder.bn.parameters())

# Define the optimizer
optimizer = torch.optim.Adam(params=params, lr=LR)

# %%
train_set.transformations = transforms.Compose([
    transforms.Resize(256),  # smaller edge of image resized to 256
    transforms.RandomCrop(224),  # get 224x224 crop from random location
    # transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),  # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))
])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, sampler=None)
train_loss_min = 100
for epoch in range(NUM_EPOCHS):
    print(f'Epoch {epoch + 1}/{NUM_EPOCHS}', flush=True)
    train_loss = train_model_new(encoder=encoder, decoder=decoder, optimizer=optimizer, loss_fn=loss_fn,
                                 train_loader=train_loader, vocab_size=vocab_size)
#     state = {
#         'epoch': epoch + 1,
#         'state_dict': final_model.state_dict(),
#         'optimizer': optimizer.state_dict()
#     }
#     if (epoch + 1) % 2 == 0:
#         torch.save(state, f'{MODEL_NAME}_ep{epoch:02d}_weights.pt')
#     if train_loss < train_loss_min:
#         train_loss_min = train_loss
#         torch.save(state, f'{MODEL_NAME}''_best_train.pt')
# torch.save(final_model, f'{MODEL_NAME}_ep{5:02d}_weights.pt')
# final_model.eval()

# %%

# model = torch.load(f'{MODEL_NAME}''_best_train.pt')
# model = final_model

# %%

try_image = train_set.imgpath_list[100]
display(Image.open(try_image))
print('Normal Max search:', greedy_predictions_gen(encoding_dict=encoding_train, model=model,
                                                   word2idx=word2idx, idx2word=idx2word,
                                                   images=train_set.images_path, max_len=max_len, device=device)(
    try_image))
for k in [3, 5, 7]:
    print(f'Beam Search, k={k}:',
          beam_search_predictions_gen(beam_index=k, encoding_dict=encoding_train, model=model,
                                      word2idx=word2idx, idx2word=idx2word,
                                      images=train_set.images_path, max_len=max_len, device=device)(try_image))

# %%

try_image = val_set.imgpath_list[4]
display(Image.open(try_image))
print('Normal Max search:', greedy_predictions_gen(encoding_dict=encoding_valid, model=model,
                                                   word2idx=word2idx, idx2word=idx2word,
                                                   images=train_set.images_path, max_len=max_len)(try_image))
for k in [3, 5, 7]:
    print(f'Beam Search, k={k}:',
          beam_search_predictions_gen(beam_index=k, encoding_dict=encoding_valid, model=model,
                                      word2idx=word2idx, idx2word=idx2word,
                                      images=train_set.images_path, max_len=max_len)(try_image))

# %%

try_image = test_set.imgpath_list[4]
display(Image.open(try_image))
print('Normal Max search:', greedy_predictions_gen(encoding_dict=encoding_test, model=model,
                                                   word2idx=word2idx, idx2word=idx2word,
                                                   images=train_set.images_path, max_len=max_len)(try_image))
for k in [3, 5, 7]:
    print(f'Beam Search, k={k}:',
          beam_search_predictions_gen(beam_index=k, encoding_dict=encoding_test, model=model,
                                      word2idx=word2idx, idx2word=idx2word,
                                      images=train_set.images_path, max_len=max_len)(try_image))

# %%

print("BLEU Scores:")
print("\tTrain")
print_eval_metrics(img_cap_dict=train_d, encoding_dict=encoding_train, model=model,
                   word2idx=word2idx, idx2word=idx2word,
                   images=train_set.images_path, max_len=max_len)
print("\tValidation")
print_eval_metrics(img_cap_dict=val_d, encoding_dict=encoding_valid, model=model,
                   word2idx=word2idx, idx2word=idx2word,
                   images=train_set.images_path, max_len=max_len)
print("\tTest")
print_eval_metrics(img_cap_dict=test_d, encoding_dict=encoding_test, model=model,
                   word2idx=word2idx, idx2word=idx2word,
                   images=train_set.images_path, max_len=max_len)
