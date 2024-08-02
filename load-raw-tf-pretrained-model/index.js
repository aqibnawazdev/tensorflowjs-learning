const MODEL_Path = "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/sqftToPropertyPrice/model.json"

let model = undefined

async function loadModel() {
  model = await tf.loadLayersModel(MODEL_Path)
  model.summary()

  //create a batch of 1
  const input = tf.tensor2d([[870]])

  //create a batch of 3
  const inputBatch = tf.tensor2d([[500], [1100], [970]])

  const result = model.predict(input)
  const resultBatch = model.predict(inputBatch)

  result.print()
  resultBatch.print() // can also use .array()

  input.dispose()
  inputBatch.dispose()
  result.dispose()
  resultBatch.dispose()
  model.dispose()
}

loadModel()