const path = require("path");
const  llama = require(path.join(
  __dirname,
  "../../build/Release/llama-addon"
));

const initResult = llama.init({
  model: '/Users/kache/models/llama/ggml-v3-13b-hermes-q5_1.bin',
  prompt: "## Instructions \n Say hello!",
});

console.log(initResult);

const result = llama.startAsync({
  model: 'foo',
  prompt: "## Instructions \n Say hello!",
  callback: (msg) => {
    console.log(`\nprint from node: ${msg}`);
  }
});