# addon

This is an addon demo that can run llama inferences in a node environment, based on [cmake-js](https://github.com/cmake-js/cmake-js).
It can be used as a reference for using the llama.cpp project in other node projects.

## Install

```shell
npm install
```

## Compile

Make sure it is in the project root directory and compiled with make-js.

```shell
npx cmake-js compile -T llama-addon -B Release
```

Passing options
```shell
npx cmake-js compile --CDLLAMA_CUBLAS="ON" -T llama-addon -B Release
```

## Run

```shell
cd examples/addon.node
node index.js
```

This is a simple demo. And, this branch is a work in progress.
Figure it out!
I'll eventually get this in.
Credits to the creator of whisper's addon.node, I mostly cargo'd the cmake strat from there. Thank you!