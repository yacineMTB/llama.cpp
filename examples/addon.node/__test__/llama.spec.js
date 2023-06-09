const path = require("path");
const llama = require(path.join(
  __dirname,
  "../../../build/Release/llama-addon"
));


describe("Run llama.node", () => {
  // TODO: unfuck
  test("it should async things", async () => {
    const result = await llama.startAsync({
      model: '/Users/kache/Downloads/OpenAssistant-SFT-7-Llama-30B.ggmlv3.q4_0.bin',
      prompt: "<|prompter|>Hello! How are you doing?<|endoftext|><|assistant|>",
      callback: (msg) => {
        console.log(msg);
      }
    })
  });
  // test("it should receive a non-empty value", async () => {
  //   llama.init({
  //     model: '/Users/kache/Downloads/OpenAssistant-SFT-7-Llama-30B.ggmlv3.q4_0.bin',
  //     prompt: "<|prompter|>Hello! How are you doing?<|endoftext|><|assistant|>",
  //   });
  //   expect(1).toBeGreaterThan(0);
  // }, 10000);
});

