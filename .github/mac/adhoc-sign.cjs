// afterPack hook ของ build Mac เท่านั้น (electron-builder-mac.json) — ฝั่ง Windows ไม่ได้เรียกไฟล์นี้
// ไม่มีบัญชี Apple Developer → sign แบบ ad-hoc ("-") ให้ทั้ง bundle
// เหตุผล: electron-builder เปลี่ยนชื่อแอป/แก้ Info.plist หลังแตก Electron → ลายเซ็นเดิมของ Electron ใช้ไม่ได้
// และ Mac ชิป Apple (arm64) ไม่ยอมรันไฟล์ที่ไม่มีลายเซ็นเลย (ขึ้นว่าแอป "damaged")
const { execFileSync } = require("node:child_process");
const path = require("node:path");

exports.default = async function adhocSign(context) {
  if (context.electronPlatformName !== "darwin") return;
  const appPath = path.join(context.appOutDir, `${context.packager.appInfo.productFilename}.app`);
  execFileSync("codesign", ["--force", "--deep", "--sign", "-", appPath], { stdio: "inherit" });
  execFileSync("codesign", ["--verify", "--deep", "--strict", appPath], { stdio: "inherit" });
};
