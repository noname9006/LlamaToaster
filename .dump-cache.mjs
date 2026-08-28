import Database from "better-sqlite3";
const db = new Database("F:/LlamaToaster/models/.cache/local-model-cache.sqlite", { readonly: true });
const rows = db.prepare("SELECT path, state, size, sha256, hf_model_id, n_layer, quant, param_count, gguf_checked_at FROM local_model_cache ORDER BY path").all();
console.log("total entries:", rows.length);
for (const r of rows) {
  console.log(`${r.state.padEnd(10)} | ${r.path.padEnd(50)} | q=${r.quant ?? "-"} | n_layer=${r.n_layer ?? "-"} | params=${r.param_count != null ? Math.round(r.param_count / 1e9 * 10) / 10 + "B" : "-"} | hf=${r.hf_model_id ?? "-"}`);
}
db.close();