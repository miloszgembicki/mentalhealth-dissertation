const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const natural = require('natural');

const maxSequenceLength = 4; // Define the maximum sequence length
const trainingData = JSON.parse(fs.readFileSync('training-data.json'));

async function testModel() {
    // Load the saved model
    const model = await tf.loadLayersModel('file://model/model.json');

    // Load the vocab object from vocab.json
    const vocab = JSON.parse(fs.readFileSync('vocab.json'));

    // Get inputKeywords from the user
    const readline = require('readline').createInterface({
        input: process.stdin,
        output: process.stdout
    });

    readline.question('Enter the sentence containing input keywords: ', input => {
        readline.close();

        // Preprocess the input data
        const tokenizer = new natural.RegexpTokenizer({ pattern: /[\s,.!?']+/ });
        const inputKeywords = tokenizer.tokenize(input);
        console.log(inputKeywords);

        // Check if all input keywords are not in vocab
        const noKeywordsInVocab = inputKeywords.every(keyword => !(keyword in vocab));

        if (noKeywordsInVocab) {
            console.log("Not sure");
            return;
        }

        const inputArray = [];
        for (let i = 0; i < inputKeywords.length; i++) {
            const keyword = inputKeywords[i];
            inputArray.push(vocab[keyword] || 0);
        }
        const filteredArray = inputArray.filter(value => value > 0);
        console.log(filteredArray);

        // Create a new array with length of maxSequenceLength, filled with zeros
        let newArray = new Array(maxSequenceLength).fill(0);

        // Insert the values from filteredArray at the beginning of the newArray
        for (let i = 0; i < filteredArray.length; i++) {
            newArray[i] = filteredArray[i];
        }

        // Create the inputTensor using the newArray
        const inputTensor = tf.tensor2d([newArray], [1, maxSequenceLength]);

        // Make predictions using the model
        const prediction = model.predict(inputTensor);

        const outputIndex = tf.argMax(prediction, axis = -1).dataSync()[0];
        if (outputIndex < trainingData.length) {
            const outputCategory = trainingData[outputIndex].output;
            console.log(`The model predicts the output category as: ${outputCategory}`);
        } else {
            console.log("Not sure");
        }
    });
}

testModel();