const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const natural = require('natural');

const trainingData = JSON.parse(fs.readFileSync('training-data.json'));

// Create vocabulary
const tokenizer = new natural.RegexpTokenizer({ pattern: /[\s,.']+/ });
const vocab = {};
let index = 1; // start index at 1 to reserve 0 for padding
for (const data of trainingData) {
    const words = tokenizer.tokenize((data.keywords || []).join(' ').toLowerCase());
    for (const word of words) {
        if (!vocab[word]) {
            vocab[word] = index;
            index++;
        }
    }
}

// Assign index to output
for (const data of trainingData) {
    const outputIndex = trainingData.findIndex(d => d.output === data.output);
    data.outputIndex = outputIndex;
}

const maxSequenceLength = 512;
const X = [];
const y = [];
for (const data of trainingData) {
    const input = data.keywords || [];
    const output = data.output;
    const inputArray = [];
    for (let i = 0; i < maxSequenceLength; i++) {
        const keyword = input[i];
        inputArray.push(vocab[keyword] || 0);
    }
    X.push(inputArray);
    y.push(output);
}

// Write vocab to JSON file
fs.writeFileSync('vocab.json', JSON.stringify(vocab));
console.log('Vocabulary saved successfully.');

const model = tf.sequential();

async function compileModel() {
    // Define model architecture
    model.add(tf.layers.embedding({
        inputDim: Object.keys(vocab).length + 1, // add 1 to inputDim for padding
        outputDim: 16,
        inputLength: maxSequenceLength, // replace with input.length
    }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({
        units: 128,
        activation: 'relu',
    }));
    model.add(tf.layers.dropout({
        rate: 0.2,
    }));
    model.add(tf.layers.dense({
        units: trainingData.length,
        activation: 'softmax',
    }));

    // Compile model
    await model.compile({
        optimizer: tf.train.adam(),
        loss: 'sparseCategoricalCrossentropy',
        metrics: ['accuracy'],
        trainable: true,
    });

    // Check if the model is compiled
    if (model.optimizer && model.loss && model.metrics) {
        console.log('Model is compiled!');
    } else {
        console.log('Model is not compiled.');
    }

    return model;
}

// Train model
async function trainModel(model) {
    await model.fit(tf.tensor2d(X, [X.length, maxSequenceLength]), tf.tensor1d(y.map(output => trainingData.findIndex(d => d.output === output))), {
        epochs: 200,
        batchSize: 64,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch + 1} / 300: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`);
            },
        },
    });
    console.log('Model trained successfully.');
}

// Save model to disk
async function saveModel() {
    await model.save('file://./model');
    console.log('Model saved successfully.');
}

compileModel().then(model => {
    trainModel(model).then(() => {
        saveModel();
    }).catch((err) => {
        console.error(err);
    });
}).catch(error => {
    console.error('Error compiling model:', error);
});