const MODEL_PATH = "https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4"
let movenet = undefined

async function loadAndRunModel() {
  movenet = await tf.loadGraphModel(MODEL_PATH, { fromTFHub: true })
  let exampleTensorInput = tf.zeros([1, 192, 192, 3], 'int32')
  let tensorOutput = movenet.predict(exampleTensorInput)

  let arrayOutput = await tensorOutput.array()

  console.log(arrayOutput)
}

loadAndRunModel()