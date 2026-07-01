# Emo Todo (web)

A tiny todo-list web app that predicts an emoji for each task on-device with
[`@desert-ant-labs/emo`](https://www.npmjs.com/package/@desert-ant-labs/emo): type
a task, watch the emoji update live, save, and click the circle to complete it.

Everything runs in the browser; the model (~5.0 MB) is fetched once from the
Hugging Face Hub and cached in Cache Storage, then works offline.

## Run

From this folder:

```bash
npm start          # or: node server.js
```

Then open <http://localhost:5173>. Pass a port as the first argument to change it
(`node server.js 8080`). There's no install or build step; the library is loaded
from npm at runtime, so an internet connection is needed on first load.

## How it works

- `package.json`: marks the example as an ES module and adds an `npm start` script.
- `server.js`: a zero-dependency Node static server for `public/`.
- `public/index.html`: loads `@desert-ant-labs/emo` from npm (via the esm.sh CDN)
  using an [import map](https://developer.mozilla.org/docs/Web/HTML/Element/script/type/importmap).
- `public/app.js`: loads the model with `load()`, debounces predictions while
  typing, and stores todos in `localStorage`.
- `public/styles.css`: the UI.
