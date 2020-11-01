import tensorflow as tf

# 1. define the data
raw_inputs = [
    [711, 632, 71],
    [73, 8, 3215, 55, 927],
    [83, 91, 1, 645, 1253, 927],
]

# 2. pad the input with post mode
padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(
    raw_inputs, padding="post", maxlen=4, truncating="post"
)

print(padded_inputs)


# 3. generate mask data
raw_length = [len(raw_inputs[i]) for i in range(len(raw_inputs))]
mask = tf.sequence_mask(raw_length, maxlen=10)
print(mask)


# 4. tensor data
raw_inputs = tf.constant(raw_inputs)
