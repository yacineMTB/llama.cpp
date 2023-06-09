const path = require("path");
const  llama = require(path.join(
  __dirname,
  "../../build/Release/llama-addon"
));

const initResult = llama.init({
  model: '/Users/kache/models/llama/ggml-v3-13b-hermes-q5_1.bin',
});

// Instructions: make this an async function that resolves when end of text event is received
const llamaInvoke = (prompt) => {
  const tokens = [];
  llama.startAsync({
    prompt,
    eventListener: (token) => {
      console.log(token);
      tokens.push(token);
      if (token === '[End of text]') {
        // Resolve the promise with all of the tokens
      }
    }
  });
  console.log(tokens);
};



const result = llama.startAsync({
  model: 'foo',
  prompt: "## Instructions \n Say hello!",
  callback: (msg) => {
    console.log(`\nprint from node: ${msg}`);
  }
});