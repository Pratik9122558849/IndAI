import json
import tensorflow as tf
import sentencepiece as spm
import os
from tqdm import tqdm

#############################################
# CONFIG
#############################################

VOCAB_SIZE = 16000
SEQ_LEN = 64
BATCH_SIZE = 32
EPOCHS = 5
EMBED_DIM = 256
NUM_HEADS = 8
FF_DIM = 512
NUM_LAYERS = 4

DATASET_FILE = "dataset/indai_dataset.jsonl"
TOKENIZER_PREFIX = "tokenizer"

#############################################
# STEP 1 — Prepare training text
#############################################

print("Preparing text for tokenizer...")

with open("train_text.txt","w",encoding="utf8") as out:

    with open(DATASET_FILE) as f:
        for line in f:

            data = json.loads(line)

            out.write(data["input"]+"\n")
            out.write(data["output"]+"\n")

#############################################
# STEP 2 — Train SentencePiece tokenizer
#############################################

print("Loading tokenizer...")

if os.path.exists("tokenizer.model"):
    sp = spm.SentencePieceProcessor()
    sp.load("tokenizer.model")
else:
    print("Tokenizer model not found")

#############################################
# STEP 3 — Encode dataset
#############################################

print("Encoding dataset...")

inputs = []
targets = []

with open(DATASET_FILE) as f:

    for line in tqdm(f):

        data = json.loads(line)

        inp = sp.encode(data["input"])
        out = sp.encode(data["output"])

        inp = inp[:SEQ_LEN]
        out = out[:SEQ_LEN]

        inputs.append(inp)
        targets.append(out)

#############################################
# STEP 4 — Padding
#############################################

inputs = tf.keras.preprocessing.sequence.pad_sequences(
    inputs,maxlen=SEQ_LEN,padding="post"
)

targets = tf.keras.preprocessing.sequence.pad_sequences(
    targets,maxlen=SEQ_LEN,padding="post"
)

dataset = tf.data.Dataset.from_tensor_slices((inputs,targets))
dataset = dataset.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

#############################################
# STEP 5 — Transformer blocks
#############################################

class TransformerBlock(tf.keras.layers.Layer):

    def __init__(self,embed_dim,num_heads,ff_dim):
        super().__init__()

        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim,activation="relu"),
            tf.keras.layers.Dense(embed_dim)
        ])

        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()

    def call(self,inputs):

        attn_output = self.att(inputs,inputs)

        out1 = self.norm1(inputs+attn_output)

        ffn_output = self.ffn(out1)

        return self.norm2(out1+ffn_output)

#############################################
# STEP 6 — Build model
#############################################

inputs_layer = tf.keras.Input(shape=(SEQ_LEN,))

x = tf.keras.layers.Embedding(VOCAB_SIZE,EMBED_DIM)(inputs_layer)

for _ in range(NUM_LAYERS):
    x = TransformerBlock(EMBED_DIM,NUM_HEADS,FF_DIM)(x)

outputs = tf.keras.layers.Dense(VOCAB_SIZE,activation="softmax")(x)

model = tf.keras.Model(inputs_layer,outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

#############################################
# STEP 7 — Training
#############################################

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "llm_checkpoint",
    save_best_only=True
)

model.fit(
    dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

#############################################
# STEP 8 — Save model
#############################################

model.save("mini_llm")

#############################################
# STEP 9 — Chat function
#############################################

def generate(prompt,max_tokens=40):

    tokens = sp.encode(prompt)

    tokens = tokens[:SEQ_LEN]

    tokens = tokens + [0]*(SEQ_LEN-len(tokens))

    tokens = tf.expand_dims(tokens,0)

    preds = model.predict(tokens)

    ids = tf.argmax(preds[0],axis=-1)

    return sp.decode(ids.numpy().tolist())

#############################################
# TEST
#############################################

print(generate("hello"))