// Panel ขวา: จัดการ "คลาส" ของ dataset (เพิ่ม/ลบ/แก้ชื่อ-สี)
// Phase 2: เพิ่ม option { onPick, getActiveIdx } (backward-compatible — ไม่ส่งมา = พฤติกรรมเดิมทุกประการ)
// เพื่อให้โหมดตีกรอบ (annotate-canvas.js) คลิกแถวเพื่อตั้ง "คลาสที่กำลังเลือก" (active class) ได้
// คุยกับ src/renderer/dataset/annotations.js (single source of truth) ตรง ๆ ไม่ผ่าน state ของ app.js
import { getClasses, addClass, removeClass, updateClass, onChange } from "../dataset/annotations.js";

export function createClassPanel(container, opts = {}) {
  const { onPick, getActiveIdx, getClassCounts, getDatasetCount } = opts;

  function render() {
    container.innerHTML = "";
    const classes = getClasses();
    const counts = getClassCounts?.() || [];

    const meta = document.createElement("div");
    meta.className = "class-panel-meta";
    meta.innerHTML = `<span>${classes.length} classes</span><span>${Number(getDatasetCount?.()) || 0} annotated images</span>`;
    container.appendChild(meta);

    const list = document.createElement("div");
    list.className = "class-panel-list";

    if (classes.length === 0) {
      const empty = document.createElement("div");
      empty.className = "class-panel-empty";
      empty.textContent = "ยังไม่มีคลาส — เพิ่มคลาสด้านล่างเพื่อเริ่มทำ label";
      list.appendChild(empty);
    }

    classes.forEach((c, idx) => {
      const row = document.createElement("div");
      row.className = "class-panel-row";
      if (onPick) {
        row.classList.toggle("active", getActiveIdx?.() === idx);
        row.classList.add("class-panel-row-pickable");
        row.title = "คลิกเพื่อตั้งเป็นคลาสที่ใช้ตีกรอบ (หรือกด 1-9)";
        // อย่าเลือกตอนคลิกช่องสี/ชื่อ/ปุ่มลบ — เฉพาะพื้นที่แถวว่าง ๆ เท่านั้นที่ตั้ง active
        row.addEventListener("click", (e) => {
          if (e.target.closest("input,button")) return;
          onPick(idx);
        });
      }

      const color = document.createElement("input");
      color.type = "color";
      color.className = "class-panel-color";
      color.value = c.color;
      color.title = "สีของคลาสนี้ (ใช้ตอนวาดกรอบ)";
      color.addEventListener("input", () => updateClass(idx, { color: color.value }));

      const name = document.createElement("input");
      name.type = "text";
      name.className = "name-input class-panel-name";
      name.value = c.name;
      name.placeholder = "ชื่อคลาส";
      name.addEventListener("change", () => {
        const v = name.value.trim();
        if (v) updateClass(idx, { name: v });
        else name.value = c.name; // กันชื่อว่าง — คืนค่าเดิม
      });

      const count = document.createElement("span");
      count.className = "class-panel-count mono";
      count.textContent = `${Number(counts[idx]) || 0} images`;

      const del = document.createElement("button");
      del.type = "button";
      del.className = "btn btn-icon class-panel-del";
      del.textContent = "🗑";
      del.title = "ลบคลาสนี้ (annotation ที่ label ด้วยคลาสนี้จะถูกลบไปด้วย)";
      del.addEventListener("click", () => {
        const ok = window.confirm(`ลบคลาส "${c.name}" ?\nannotation ที่ label ไว้ด้วยคลาสนี้จะถูกลบไปด้วย`);
        if (ok) removeClass(idx);
      });

      row.append(color, name, count, del);
      list.appendChild(row);
    });

    container.appendChild(list);

    // แถวเพิ่มคลาสใหม่
    const addRow = document.createElement("div");
    addRow.className = "class-panel-add";

    const addInput = document.createElement("input");
    addInput.type = "text";
    addInput.className = "name-input class-panel-name";
    addInput.placeholder = "Class name";

    const addBtn = document.createElement("button");
    addBtn.type = "button";
    addBtn.className = "btn class-panel-add-btn";
    addBtn.textContent = "+ Add";

    function submitAdd() {
      const v = addInput.value.trim();
      if (!v) return;
      addClass(v);
      addInput.value = "";
      addInput.focus();
    }
    addBtn.addEventListener("click", submitAdd);
    addInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") submitAdd();
    });

    addRow.append(addInput, addBtn);
    container.appendChild(addRow);
  }

  render();
  // subscribe ครั้งเดียวตอนสร้าง panel (ไม่ subscribe ซ้ำใน render) — กัน listener ค้างเวลา re-render บ่อย
  const unsubscribe = onChange(render);

  return {
    render,
    destroy: () => unsubscribe(),
  };
}
