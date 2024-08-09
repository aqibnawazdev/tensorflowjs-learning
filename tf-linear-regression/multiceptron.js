import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/real-estate-data.js"

//Input feature pairs (House size, Number of Bedrooms)
const INPUTS = TRAINING_DATA.inputs
//Currently listed house prices in dollars given their features above (target output values you want to predict)
const OUTPUTS = TRAINING_DATA.outputs

//Shuffle the two arrays in the same way so inputs still match outputs indexes
tf.util.shuffleCombo(INPUTS, OUTPUTS)

//Input feature Array of Array need 2D tensor to store.
const INPUTS_TESNOR = tf.tensor2d(INPUTS)

//Output can stay 1 diamensional
const OUTPUTS_TESNOR = tf.tensor1d(OUTPUTS)

//Funtion to take a tensor and normalize values with respect 
//to each column of values contained in that Tensor.
function normalize(tensor, min, max) {

  const result = tf.tidy(function () {
    //Find the minimum vlaue contained in tensor
    const MIN_VALUES = min || tf.min(tensor, 0)

    //Find the maximum value contained in tesnor
    const MAX_VALUES = max || tf.max(tensor, 0)

    //Now subtract MIN_VALUES form every value in Tensor  
    // and store the value in new Tensor

    const TENSOR_SUB_MIN_VALUE = tf.sub(tensor, MIN_VALUES)

    //Calculate the range size of possible values.
    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);

    //Calculate the adjusted value divided by the range size as  new Tensor
    const NORMALIZED_VALUES = tf.div(TENSOR_SUB_MIN_VALUE, RANGE_SIZE)

    return { NORMALIZED_VALUES, MIN_VALUES, MAX_VALUES }
  })

  return result

}

const FEATURE_RESULTS = normalize(INPUTS_TESNOR)
console.log("Normalized values... ")
FEATURE_RESULTS.NORMALIZED_VALUES.print()

console.log("Min values... ")
FEATURE_RESULTS.MIN_VALUES.print()

console.log("Max values... ")
FEATURE_RESULTS.MAX_VALUES.print()


INPUTS_TESNOR.dispose()

//Now actually create and define model architecture
const model = tf.sequential();

//We will use one dense layer with 100 neurons (units) and an input of 
//1 input feature values 

model.add(tf.layers.dense({ inputShape: [1], units: 25, activation: 'relu' }))

//Add new hidden layer with 100 neurons, and ReLU activation
model.add(tf.layers.dense({ units: 5, activation: 'relu' }))

//Add another dense layer  with 1 neuron that will be connected to previous hidden layer.
model.add(tf.layers.dense({ units: 1 }))

model.summary()

async function train() {
  const LEARNING_RATE = 0.00001 //Choose learning rate that's suitable for the data we are using.

  //Compile the model with the defined learning rate and specify a loss function to use.
  model.compile(
    {
      optimizer: tf.train.sgd(LEARNING_RATE),
      loss: 'meanSquaredError'
    }
  )
  // finally do the training itself

  let results = await model.fit(FEATURE_RESULTS.NORMALIZED_VALUES, OUTPUTS_TESNOR, {
    validationSplit: 0.15, //Take aside 15% of the data for validation testing.
    shuffle: true, //Ensure data is shuffled in case it was in an order
    batchSize: 64, //As we have a lot of training data, batch size is set to 64.
    epochs: 10 //Go over the data for 10 times
  })

  OUTPUTS_TESNOR.dispose()
  FEATURE_RESULTS.NORMALIZED_VALUES.dispose()

  console.log("Avg error loss: " + Math.sqrt(results.history.loss[results.history.loss.length - 1]));
  console.log("Average validation error loss: " + Math.sqrt(results.history.val_loss[results.history.val_loss.length - 1]))
  evaluate() //Once trained evaluate the model
}
train()

function evaluate() {
  //Predict answer for single piece of data

  tf.tidy(function () {
    let newInput = normalize(tf.tensor2d([[750, 1]]), FEATURE_RESULTS.MIN_VALUES, FEATURE_RESULTS.MAX_VALUES);
    let output = model.predict(newInput.NORMALIZED_VALUES)
    output.print()
  })
  FEATURE_RESULTS.MIN_VALUES.dispose()
  FEATURE_RESULTS.MAX_VALUES.dispose()
  model.dispose()

  console.log(tf.memory().numTensors)
}