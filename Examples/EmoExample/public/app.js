// Emo Todo — a tiny browser todo list that predicts an emoji for each task
// on-device with @desert-ant-labs/emo. Mirrors the SwiftUI EmoExample app.
//
// @desert-ant-labs/emo is loaded from npm via the import map in index.html.
// The model (~3.8 MB) is fetched once from the Hugging Face Hub and cached in
// Cache Storage, then works offline.
import { load } from "@desert-ant-labs/emo";

const STORAGE_KEY = "todos";

// --- Model --------------------------------------------------------------

let emoPromise = null;

function loadEmo() {
  emoPromise ??= load();
  return emoPromise;
}

async function suggestions(text, limit) {
  const emo = await loadEmo();
  return emo.suggestions(text, limit);
}

// --- Store --------------------------------------------------------------

function makeId() {
  return crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function loadTodos() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) ?? [];
  } catch {
    return [];
  }
}

let todos = loadTodos();

function saveTodos() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(todos));
}

// --- Elements -----------------------------------------------------------

const todoList = document.getElementById("todoList");
const emptyState = document.getElementById("emptyState");
const addButton = document.getElementById("addButton");

const sheet = document.getElementById("addSheet");
const backdrop = document.getElementById("sheetBackdrop");
const cancelButton = document.getElementById("cancelButton");
const cancelButton2 = document.getElementById("cancelButton2");
const saveButton = document.getElementById("saveButton");
const titleField = document.getElementById("titleField");
const badgeEmoji = document.getElementById("badgeEmoji");

// --- List rendering -----------------------------------------------------

const CHECK_SVG = `<svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="3.4" stroke-linecap="round" stroke-linejoin="round"><path d="M5 13l4 4L19 7"/></svg>`;

function render() {
  todoList.innerHTML = "";
  emptyState.hidden = todos.length > 0;

  for (const todo of todos) {
    const row = document.createElement("li");
    row.className = "row";
    row.dataset.id = todo.id;
    row.innerHTML = `
      <span class="row-emoji">${todo.emoji}</span>
      <span class="row-title"></span>
      <button class="check" aria-label="Complete">
        <span class="check-ring"></span>
        <span class="check-fill">${CHECK_SVG}</span>
      </button>`;
    row.querySelector(".row-title").textContent = todo.title;
    row.querySelector(".check").addEventListener("click", () => complete(todo.id, row));
    todoList.appendChild(row);
  }
}

function complete(id, row) {
  if (row.classList.contains("completing")) return;
  row.classList.add("completing");
  setTimeout(() => {
    row.classList.add("removing");
    setTimeout(() => {
      todos = todos.filter((t) => t.id !== id);
      saveTodos();
      render();
    }, 300);
  }, 320);
}

// --- Add sheet ----------------------------------------------------------

let predictionTimer = null;
let predictionSeq = 0;
let currentEmoji = "✨";

// Keep the modal centered within the *visible* viewport (above the on-screen
// keyboard on mobile) using the VisualViewport API.
const vv = window.visualViewport;

function positionModal() {
  if (!vv || sheet.hidden) return;
  sheet.style.top = `${vv.offsetTop + vv.height / 2}px`;
  sheet.style.maxHeight = `${Math.max(0, vv.height - 24)}px`;
}

function openSheet() {
  titleField.value = "";
  setBadge("✨", false);
  saveButton.disabled = true;
  sheet.hidden = false;
  backdrop.hidden = false;
  document.body.classList.add("modal-open");
  positionModal();
  vv?.addEventListener("resize", positionModal);
  vv?.addEventListener("scroll", positionModal);
  requestAnimationFrame(() => {
    sheet.classList.add("show");
    backdrop.classList.add("show");
  });
  setTimeout(() => titleField.focus(), 250);
}

function closeSheet() {
  sheet.classList.remove("show");
  backdrop.classList.remove("show");
  clearTimeout(predictionTimer);
  document.body.classList.remove("modal-open");
  vv?.removeEventListener("resize", positionModal);
  vv?.removeEventListener("scroll", positionModal);
  titleField.blur();
  setTimeout(() => {
    sheet.hidden = true;
    backdrop.hidden = true;
    sheet.style.top = "";
    sheet.style.maxHeight = "";
  }, 400);
}

function setBadge(emoji, hasText) {
  currentEmoji = emoji;
  if (badgeEmoji.textContent !== emoji) {
    badgeEmoji.textContent = emoji;
    badgeEmoji.classList.remove("pop");
    void badgeEmoji.offsetWidth; // restart animation
    badgeEmoji.classList.add("pop");
  }
  badgeEmoji.classList.toggle("has-text", hasText);
}

function onTitleInput() {
  const trimmed = titleField.value.trim();
  saveButton.disabled = trimmed.length === 0;

  clearTimeout(predictionTimer);
  if (!trimmed) {
    setBadge("✨", false);
    return;
  }

  const seq = ++predictionSeq;
  predictionTimer = setTimeout(async () => {
    try {
      const next = (await suggestions(trimmed, 1))[0]?.emoji;
      if (seq === predictionSeq && next) setBadge(next, true);
    } catch {
      /* keep current emoji */
    }
  }, 200);
}

async function save() {
  const title = titleField.value.trim();
  if (!title) return;
  predictionSeq++; // invalidate pending prediction
  clearTimeout(predictionTimer);

  let emoji = currentEmoji;
  try {
    emoji = (await suggestions(title, 1))[0]?.emoji ?? currentEmoji;
  } catch {
    /* fall back to current emoji */
  }

  try {
    todos.push({ id: makeId(), title, emoji });
    saveTodos();
    render();
  } catch (err) {
    console.error("Failed to save todo", err);
  }
  closeSheet();
}

// --- Wiring -------------------------------------------------------------

addButton.addEventListener("click", openSheet);
cancelButton.addEventListener("click", closeSheet);
cancelButton2.addEventListener("click", closeSheet);
backdrop.addEventListener("click", closeSheet);
saveButton.addEventListener("click", save);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !sheet.hidden) closeSheet();
});
titleField.addEventListener("input", onTitleInput);
titleField.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    save();
  }
});

render();

// Warm up the model in the background so the first prediction is instant.
loadEmo().catch(() => {});
