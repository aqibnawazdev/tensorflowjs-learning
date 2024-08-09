const MODEL_PATH = "https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4"
let movenet = undefined
const EXAMPLE_IMG = document.getElementById("exampleImg")
async function loadAndRunModel() {
  movenet = await tf.loadGraphModel(MODEL_PATH, { fromTFHub: true })
  let exampleTensorInput = tf.zeros([1, 192, 192, 3], 'int32')

  let imageTensor = tf.browser.fromPixels(EXAMPLE_IMG)
  console.log(imageTensor.shape)

  let cropStartPoint = [15, 179, 0]
  let croptSize = [345, 345, 3]
  let cropedTensor = tf.slice(imageTensor, cropStartPoint, croptSize)

  let resizedTensor = tf.image.resizeBilinear(cropedTensor, [192, 192], true).toInt()


  console.log(resizedTensor.shape)


  let tensorOutput = movenet.predict(tf.expandDims(resizedTensor));

  let arrayOutput = await tensorOutput.array()

  console.log(arrayOutput)
}

loadAndRunModel()