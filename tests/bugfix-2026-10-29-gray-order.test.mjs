// แก้รีวิว 2026-10-29 — ลำดับช่องขาวดำ + PIQE ไม่เดินสาย B ที่ไม่ได้ใช้
import assert from "node:assert/strict";
import { test } from "node:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import crypto from "node:crypto";
import { execFileSync } from "node:child_process";
import cv from "@techstark/opencv-js";
import { OPS, FILTERS, opIdOfStep } from "../src/renderer/filters/registry.js";
import { OP_TO_FILTER, createPalettePanel } from "../src/renderer/ui/palette-panel.js";
import { currentOpId } from "../src/renderer/filters/ui-meta.js";
import { buildPythonCode } from "../src/renderer/codegen.js";
import { assertWrapperBalance } from "./_leak-helpers.mjs";

while (typeof cv.Mat !== "function") await new Promise((r) => setTimeout(r, 20));
globalThis.self = { postMessage() {} };
const { runPipeline, collectNodeIds, sweepNodeState } = await import("../src/renderer/cv/cv-worker.js");
const defaults = (id) => Object.fromEntries(OPS[id].params.map((q) => [q.key, q.default]));
function step(id, p = {}) {
  const fid = OP_TO_FILTER[id] || id, local = { ...defaults(id), ...p };
  return fid === id ? { id, params: local } : { id: fid, params: { mode: id, ...Object.fromEntries(Object.entries(local).map(([k, v]) => [id + "__" + k, v])) } };
}
function two(id, a = [], b = []) {
  return [...a, { ...step(id), src2: { pipeline: b } }];
}
const color = (space = "hsv") => step("colorspace", { space });
function fixture() {
  const data = new Uint8Array(33 * 31 * 3);
  for (let y = 0; y < 31; y++) for (let x = 0; x < 33; x++) data.set([(x * 17 + y * 3) % 256, (x * 5 + y * 23) % 256, (x * 29 + y * 11 + 91) % 256], (y * 33 + x) * 3);
  return cv.matFromArray(31, 33, cv.CV_8UC3, data);
}
const grayLines = (code) => code.split("\n").filter((s) => /cvtColor\(.*COLOR_(BGR|RGB)2GRAY/.test(s));

test("gray-order. seven app-order spaces use RGB2GRAY in explicit and automatic main conversions", () => {
  for (const space of ["hsv", "hls", "lab", "luv", "ycrcb", "xyz", "yuv"]) {
    const code = buildPythonCode([color(space), step("grayscale")]);
    assert.ok(code.includes("img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)"), space);
  }
  for (const id of ["otsu_threshold", "canny", "focus_score", "watershed"]) {
    const lines = grayLines(buildPythonCode([color(), step(id)]));
    assert.ok(lines.length > 0 && lines.every((s) => s.includes("COLOR_RGB2GRAY")), id);
  }
  assert.ok(buildPythonCode([color(), step("watershed")]).includes("img[markers == -1] = (255, 0, 0)  # เส้นแบ่งวัตถุสีแดง"));
});

test("gray-order. two-image A/B and nested B keep independent order and reset after auto-gray", () => {
  for (const id of ["logical_ops", "mask", "hist_match", "template_matching", "psnr", "ssim", "seg_score", "ecc_register"]) {
    const code = buildPythonCode(two(id, [color()], [color("lab")]));
    const lines = grayLines(code);
    assert.ok(lines.length > 0, id);
    assert.ok(lines.every((s) => s.includes("COLOR_RGB2GRAY")), id + ": " + lines.join("\n"));
  }
  const aOnly = grayLines(buildPythonCode(two("logical_ops", [color()])));
  assert.ok(aOnly.find((s) => s.includes("cvtColor(img,")).includes("COLOR_RGB2GRAY"));
  assert.ok(aOnly.find((s) => s.includes("cvtColor(img_b,")).includes("COLOR_BGR2GRAY"));
  for (const tail of ["grayscale", "canny", "frangi_vesselness"]) {
    const code = buildPythonCode(two("psnr", [], [color(), step(tail)]));
    assert.ok(grayLines(code).some((s) => s.includes("cvtColor(img_b") && s.includes("COLOR_RGB2GRAY")), tail);
  }
  const nested = buildPythonCode(two("psnr", [], two("logical_ops", [color()], [color("lab")])));
  assert.ok(nested.includes("cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)"));
  assert.ok(nested.includes("cv2.cvtColor(img_b2, cv2.COLOR_RGB2GRAY)"));
  // auto-gray ได้ภาพเทาแล้ว: สีที่วาด/การวัดภายใน op ถัดไปต้องกลับไปใช้ BGR
  const reset = grayLines(buildPythonCode(two("psnr", [color(), step("canny"), step("watershed")])));
  assert.ok(reset[0].includes("COLOR_RGB2GRAY"));
  assert.ok(reset[1].includes("COLOR_BGR2GRAY"));
});

// Generated once from git show HEAD:src/renderer/codegen.js at bcdbf4a in OS temp.
// Freeze these cases independently of future OPS additions; each pin names its export.
// rgb_adjust was re-pinned for the approved float32/ties-to-even rounding fix.
const MIRRORED_OP_IDS = [
  "grayscale", "colorspace", "channel_split", "rgb_adjust", "brightness_contrast", "gamma",
  "hist_equalize", "clahe", "contrast_stretch", "flatfield_correction", "quantize", "unsharp",
  "laplacian_sharpen", "custom_kernel", "salt_pepper_noise", "median_blur", "gaussian_blur", "box_blur",
  "bilateral", "morphology", "binary_threshold", "otsu_threshold", "multi_otsu_threshold", "adaptive_threshold",
  "canny", "sobel", "laplacian_edge", "scharr_edge", "prewitt_edge", "roberts_edge",
  "log_edge", "find_contours", "connected_components", "distance_transform", "local_std_filter", "local_entropy_filter",
  "regionprops", "watershed", "bounding_boxes", "hough_circles", "hough_lines", "harris_corners",
  "shi_tomasi_corners", "orb_keypoints", "akaze_keypoints", "fast_keypoints", "grabcut", "region_growing",
  "color_inrange", "invert", "white_balance", "kmeans_segment", "crop", "resize",
  "rotate", "flip", "perspective_warp", "undistort", "fft_magnitude", "fft_filter",
  "imabsdiff", "imfuse", "mask", "imarith", "logical_ops", "inpaint",
  "hist_match", "template_matching", "psnr", "ssim", "mog2", "farneback_optical_flow",
  "pyrlk_optical_flow", "gaussian_noise", "wiener_filter", "translate_shear", "feret", "line_profile",
  "shape_filter", "roi_polygon", "slic", "bandpass_notch", "guided_filter", "focus_score",
  "pseudocolor", "delta_e", "roi_apply", "deconv_wiener", "deconv_lucy", "ecc_register",
  "gabor_filter", "glcm", "nlm_denoise", "dehaze", "dct_jpeg", "watershed_marker",
  "seg_score", "lowlight_enhance", "colorspace_inverse", "euler_number", "image_pyramid", "frangi_vesselness",
  "anisotropic_diffusion", "debayer", "piqe", "pass_fail", "morph_erode", "morph_dilate",
  "morph_open", "morph_close", "morph_gradient", "morph_tophat", "morph_blackhat", "morph_fill_holes",
  "morph_clear_border", "morph_remove_small", "morph_clean", "morph_majority", "morph_skeleton", "morph_spur",
  "morph_thin", "morph_shrink", "morph_reconstruct_open", "morph_convex_hull", "morph_endpoints_branchpoints", "morph_hit_or_miss",
  "morph_regional_max", "morph_regional_min", "morph_h_maxima", "morph_h_minima", "morph_extended_max", "morph_extended_min",
];
const MIRRORED_SHA256 = {
  "grayscale": "7e91be9f9d662f3b34c90c5ebba127dd7990c33ed7d1b615d9e213e9398269ee",
  "colorspace": "c8fb8ab0542a133b25e2a2b9b4d73b3dd78bc88a944842333dbae1182efa8ae6",
  "channel_split": "6467ea211f2682a886e4a34d54cb81443c8548e56c261095d0a67902844b2dd5",
  "rgb_adjust": "947698183e82cbaf406c176c7cc12cf3c9b1682e48e6f1a33ce5cc87d8b66bd9",
  "brightness_contrast": "b808d21deb1afaa13f36007cdb4d32fdf921b89ac3b74599caee67f303bc256b",
  "gamma": "eaac661bfd0112d1fb877d5891945dc602c20e26e2310309d89dfece88dd160c",
  "hist_equalize": "d583c62ee161b2e7096b9869c8643021078b8e72e5f73488ba4786e5f8d231cb",
  "clahe": "c3ac2f05d472a13b5ef253ef1ca54e42c2cfb09294efb60bcf091891da09c08d",
  "contrast_stretch": "e73891772ae32782c413b58cc6afde2cc9f3c43b388d6ef83307534601fc1632",
  "flatfield_correction": "a1c291b03feb984d15a0863e75ff2f6ac7bd1fdde2a4985a9d2bf27273da3e11",
  "quantize": "fd6dd6ae655ba131834aa461832ad7ecf19899e6eba05b6cf872804c5bf977a6",
  "unsharp": "77864dd8e2f875c9c674313d8733bf7fddd659feeb1dc2fd5271a3f7fe53ea88",
  "laplacian_sharpen": "3b90c1abb7a9eff04304760651bf002b5900c827ee8ba851ac50e81167102948",
  "custom_kernel": "6b6c50a85860fbf0e83fcc1a285d60b0424c792355774820cf8f530bf2d5f7ca",
  "salt_pepper_noise": "4c0788ab1a2a996b8e11731a1df9746a795a0dc2fadd6c4cdce08213874dc27d",
  "median_blur": "893125e13ee898b4d384e8577a680ba0b3aa946217a15db267a222ea7df4b2a9",
  "gaussian_blur": "891f72b00ba690057f280e42c638287c19ccf95c6f3cc50502fab068d1db7b68",
  "box_blur": "d4dd89cac1b433a99a5f5015e7e07b345cab46192b0c8e408bfaaa075d91baf1",
  "bilateral": "f9968c1e89731d6916be1517b1ece97bdce4bc15025bd5b7ebc4098208ca0ec0",
  "morphology": "0478779123a38da0b3b11fd2756dc911cae8374cbf561caef7a94ec29d692301",
  "binary_threshold": "9e04f92ca87c9eaf080614dc62a21c73590c384858ace23ab869e070880cebc8",
  "otsu_threshold": "7256fe7e5a863361cb5951b930322d6065502c647487e1fcdc77c2ea147463ab",
  "multi_otsu_threshold": "63c228b9c32def1bda231a7b007c4eddfae063d2c07ad54e903f56423158e056",
  "adaptive_threshold": "670497c079d7df0ecc7fedff76bb2877f95a9b2d2be1b174471865186172b28b",
  "canny": "cfb129a71e8ddc0a106263a98d00754220c34022fbc712db243338091fd4f6af",
  "sobel": "d39c0d02cec5e7a008b9bfde9b29fa53000ba3a5820d08b5962b3bedbb6921af",
  "laplacian_edge": "571f5ecee2ccd3a05bb799f133a653486fd80e2b82e1520919619d1b88076b89",
  "scharr_edge": "5b32d67b8ad77f087157862564cf22f85f4441a22850583696857af746761648",
  "prewitt_edge": "562ce627c39b727943be4ac7764f3b8f94de7023c7faf9f622a995d320831387",
  "roberts_edge": "de95330be754a0c1a4d232631f53dc83160e0ecc35ce22a8dcc07d64d1cfb62a",
  "log_edge": "3c17055f490a47973fe90dc1518b43f2849c10f77273b0a3cad2c8335b02fa09",
  "find_contours": "23f5f45ea4052645b837a13ebc2af388494309ffa31e69cd8239b91159469465",
  "connected_components": "d54ffe07fce06e64649e3a7384963a17ae8210c90607fb80d01da81399bddd84",
  "distance_transform": "1886aa1ccab89a9e03057519a6c43ef6d6f23af275fb906288078075daa50d33",
  "local_std_filter": "28824ea969adb383da8e6f95130a02109d81a6ee98373f5e7d69341611bf8f5a",
  "local_entropy_filter": "f342e4a5abda26e3727ce5f8f9b9814261e7ac875f7853779ad0f124c394c84f",
  "regionprops": "41b5895ba0c9b6cc45e1bd53bd63932eaa42c4a4ad0b4cc6b98c9b5c11067318",
  "watershed": "61f72f3f2470c7c915ed290cb031ab54499bba094c5ed98e31b0a84d35ca2521",
  "bounding_boxes": "36552bc96cf40f7e91457f92a8210b5a25b804c27bcd2b0b28a35eadefbac95d",
  "hough_circles": "1ea0e248075e8af24afcbaaad535bd081d7d91fb79a3cff0e4a1e66212d32280",
  "hough_lines": "7fdfcf448ff5813f71e206e939439b430749f74ad5457e0c8db2fdd94afba2d2",
  "harris_corners": "556e4d1923d7b60e7c3743e4a7b57710f4f0460d367f96f18bb80a0ed1b50a19",
  "shi_tomasi_corners": "66cda269b9d05cb6a534edcf22873bece05ad5351ef823d6bf0b3abdaebe4822",
  "orb_keypoints": "61d8dd5f3d67414ece5a9d654f53658fedd463f8d9917bf9497db94556153b78",
  "akaze_keypoints": "e20139db9489a104b4ad8347a8925a29153deeb3cb9ee838e74531e42775b6a9",
  "fast_keypoints": "5ce0a71358598ea2de16039dbe6b06e423c607d8c82a5fff0ac8461435ba1761",
  "grabcut": "e64c2c9e4d5dac3453006e139f834cb67a045852634161f5615c7d11627c8e0d",
  "region_growing": "c86086fa7169182183b0fcd1c012ed873de19e18bef890b8877ae133b7597d38",
  "color_inrange": "609a55c798b20e7f3e0c61c774653dc5732b420fd0ea671d5865e844075114aa",
  "invert": "f54d48faccf44064670fef450024216f1912e41813368439b66f6797016e08fa",
  "white_balance": "26dbfc104d7cd784b04a48060bb4b453d983f00a8c7c682d439b35e11aa55a50",
  "kmeans_segment": "a52a499d3300f7d94bb6c0c519760a36bb9b986dfead9548de6ed3fa584fabd0",
  "crop": "0592f3e90a3830d573a483b0383984b3c99717b430cfa2135f71f9526d6cc617",
  "resize": "e2f3140f5955bc874c801a615baa5791a900ae6ad930617e7abaae2bab107cef",
  "rotate": "cfb894ca640c15c3ec5737986b37057b6cef2818220682f267c0a6098d5b750a",
  "flip": "9d8e30e211c1611a779b8ced638141cf0d1d7513c41bd4302ce0e27866289854",
  "perspective_warp": "f2fc345e5b7b50195ea651755482fcb6296f36ca1379e2f139fbf1b233cfa935",
  "undistort": "1fdf2114469d604350a75e5268e0edd10ef17bd83ad557d9f14300f3d4314c01",
  "fft_magnitude": "81e9d7a2885f605add5c2bca4ca5bffd66f56b808a44d245721b2b0281644980",
  "fft_filter": "3c413265bae7bb5c6e87852c28c0c0d206e1a523c5a7bada3a485716836c7f1a",
  "imabsdiff": "69dfae2e8e3bb15b2171baa51d9bed9780bd254114060cb2e0f0f007c28c613e",
  "imfuse": "138bbba73bd2e4b03bdbcb53a9eca8096ee425a107889c299724b76a9d4bf86f",
  "mask": "0970fd0d0245d9f40ff3105763594a6ea6945e5fe658f96f8622a0f839f6588f",
  "imarith": "c1d4e670a63a7200bdf398c1a964c1cd3f5145629aaee9fc22735cd6f130be90",
  "logical_ops": "2760a9b43a86e285c0b0c372d3e0d4a7b77d5f1473c52fb8b2a1b5e0498c3fc6",
  "inpaint": "219547e48f501f0e54e6ce5228943ec4861bfba604d874e8d0b13c185dbb8328",
  "hist_match": "64d783f2a30f7549cf457e35167e771c83c6e96ab681bb2ead735f0815d669ec",
  "template_matching": "eb8c588c3756c1d8484af0628e9d094a25ba9641b661f135460879253969f792",
  "psnr": "352e1a2992f509e2b055761119d8a1983295ad7e71aba96ba27076edec6d78bc",
  "ssim": "6d11aff011c21c2131a367e8cf2263f7136e2ff37fe4087a6f9d38f2b6b28b40",
  "mog2": "1687ec87a327f1668f05884351f8e0c96641b7e143c85dc5d8ab9a970c9d1105",
  "farneback_optical_flow": "2528cb981efedc3a159746e9f80524ca52c3aa7e1841866c4a9c0f24b7b2163d",
  "pyrlk_optical_flow": "fa669d3268e7fa5e6c33b42938a7880d92fce811d69154f41d4b4a6525f610e2",
  "gaussian_noise": "3b65a38a9b137281b8223f44d02436ea347d82a33216b252f56c62a374dec760",
  "wiener_filter": "1850afb77476eb12af9000366c5949ed0136ba694ff3277c63697478cb18a831",
  "translate_shear": "4952f8d08895b7e874a47eb6ed6ca7b08449f1179205a915de47a4ff4d0802da",
  "feret": "f429ee07c3834afec10c6bac158caa370eb86b04608d0157fa2b9bda5073848a",
  "line_profile": "e0838d74aa136acff259c9d57b69bf65a95e85c769dbf0817690cdaec10998fd",
  "shape_filter": "e68faa161ee1176721f8e866fec02a380600dc9c35561f8a3d336daa3d1df041",
  "roi_polygon": "146ed683f7892640395e2e3a76a17556672fda160bf546429b6b343a40f58a0d",
  "slic": "9b877c4f9a3bb922579db30bda7d452ed645c8a16368736d60a627c6a62de2c9",
  "bandpass_notch": "9a17a8bf183ea63e2189dedf071daf634c8472615c818549d6ac42cf06060ccc",
  "guided_filter": "4fe3d0a5da841d0f1547fb542826129be2cba3d419a3d3cc72d529fdd5a1d633",
  "focus_score": "396d5dc5ab15f45313cb3f503f493d53a895df60087eaa352ff76600fec47457",
  "pseudocolor": "8ab1d77845cee2479f37d0c516c198fc6efedb1919cf05c52014a54f75bc00cb",
  "delta_e": "59b09bd7999b2a3924ff2d4cd94a38b73d1bd68439da3a6f842fe1efc16a6ae5",
  "roi_apply": "3eb2b42a27656c021560b0dcb7df5a5db7ef2f048a2da737aee5b1e439662a34",
  "deconv_wiener": "020e201ac19ba5cae59dc5a12de4f4bb153624f0317594b9b97ae31bda57e24e",
  "deconv_lucy": "69371b23487e2563f3abded21fecddfeba8239629e25a5089bccf8cf9e57070b",
  "ecc_register": "7d42a6d62933e4613192a5204b0d278323b53874cc9f92a2710de342330f59dc",
  "gabor_filter": "989dcbb0da8602284eb57fba7843ce65eba75869e48f08345d2c698c3f937251",
  "glcm": "aa3274693cf4b923467c5c7c26f3326c38e19b56a6ce9d4d04841e03546affa1",
  "nlm_denoise": "286041265cadb905277882bb602984e84673e022c1fffad992bd0328de4a5ceb",
  "dehaze": "dd52a5971743de13ad903040512a23a0fb4771ab278d27a984185dd891d21283",
  "dct_jpeg": "9786de253355f0ebed4503a931b4ecb36bfa2ffc15f87ef5e766ff4c0a348b5a",
  "watershed_marker": "f4f9a8973f3f459c7c362fa5c6d485919d611e0bd6145ddb3cdeb6c7b1422645",
  "seg_score": "b6fc848ad7a7502da980196decd3b8ecd92a3a0831e48dccc7a9b15d64e3a821",
  "lowlight_enhance": "2012647d099570f649460f7a545e72d69e37f54ec2be6cc23035865a637e53f9",
  "colorspace_inverse": "21f3f43d409b3f0804c147f92ed0329a5c00c050ecd3d429729047c6cdd55226",
  "euler_number": "9178a5f713afe97ba01f1148c00e34189aa53d6f766f3cfb1bee34ae468dfb7b",
  "image_pyramid": "ccd2ca2e05504786b59783d775cba362f0f53bac1e96d0127ef4963386a48fec",
  "frangi_vesselness": "adfdcfedf785c8c0f8d4ed5ff5ac867ba49a97e0ca888ac333626f938f5108ba",
  "anisotropic_diffusion": "85ce64867b9532eccdd14f4d686efc5ea2b4af80075d4a3eb28bce9f53ee7e0d",
  "debayer": "4483d51ac6768a0ace9c42e055d36ce4279bf3a716c7c28f2952d5d02a0c013d",
  "piqe": "4c9228b8c1f5c78db11bab9e134b5cb8415ac14322cb858a4c12a538ff9164df",
  "pass_fail": "e270304893f62a5f79329a795030d035fcbe663bf155c0f676de9ab7a79db40c",
  "morph_erode": "4788a799d6ed90d6288db9a6a25e242f9f6bc167955d3b6cb1106beed4860ed6",
  "morph_dilate": "5808dd92d035cebc2d59f65b7e8611c14902211dbd7238265398a7d5bb249f37",
  "morph_open": "6a29c1c5b098824e841a44f0e76130600f2b3b1b9cac914ce1c7eaf1ee8fa562",
  "morph_close": "bf7c074ba02fafe3b7abe0e56ca439c3ad3f1b6bbf764d3d3a60db232292aa9e",
  "morph_gradient": "9be501731c1a9d38da37a20dc58e6701b6d6d412a0e0542c07db8a15481a6b1b",
  "morph_tophat": "14e6d67416936ea5ea55b5c7af0c1d7a197b5902fb843ecd9b25429bc572ee4d",
  "morph_blackhat": "48fb793051fa0bb0ccef5f5e02064b8b0f76a2dfe2639f757680a9a6f824cde8",
  "morph_fill_holes": "f6cb359822742db4f23412cec93e277a54c98356fe17135d9a6f68f749032f06",
  "morph_clear_border": "f2e05fe6abaf0c1e32ba72f8f48a4e2538911c4d0b98af5a60f26dd5638d1a20",
  "morph_remove_small": "e6e9233c3beeb3998ccd1fa9d9964a92e15821e3b8f56f44ccc7040060907a3f",
  "morph_clean": "793e7edd1531794d70bc72fdd8180da448ede15d1da1cc29a165a09428222940",
  "morph_majority": "87c6b8c0f01aff32c7407be8c1e4ab7b55b12450ba20739a21e4a382778f1737",
  "morph_skeleton": "0d9886e3749224205e4f2d518dc1b3efdf6039af40df4772729c14acf4100b45",
  "morph_spur": "e0ace57237d887775a1b558fe91ebbd8886a6d6d05d1be1e4170c0757d8b218d",
  "morph_thin": "64a6f94280cacc75dba89235d8de23b568cfc82ac344d175d304b944fcca2a58",
  "morph_shrink": "5235de03743734d33a25d8e371db64bc1f3e9af048555bfbba9c10e57e2e22b3",
  "morph_reconstruct_open": "740fdd2c16fc3f78e3894879dd3e7939caeae7de34a81de43b5f44108de22dea",
  "morph_convex_hull": "c471a349efa7432f04c17d578b474c21ab71d176a7d2da94f4faa49e994eb30d",
  "morph_endpoints_branchpoints": "85f1ca9ea56acb0e1f9e9f2c2996d6367ac179e5055a0209951aff7336dfbb60",
  "morph_hit_or_miss": "31359fc36c151cdba45f5785cfa3b7de50d52f386a70995bd7f970f81eda48d2",
  "morph_regional_max": "3df7f24ae4813a3c09025c4f8e62b4203133662f55ed0f23de02ac81374213d5",
  "morph_regional_min": "4b43c652808c40c90c17c971a1e7797f9b2372053c72f428909ea64531229769",
  "morph_h_maxima": "9ed939eb9e0e2ce4d4cd3f2c9534d7548de24b38a466a3d50bba3ae19351037d",
  "morph_h_minima": "e76d26e21033421ef7bfccae824e1da0039218c048dde905c9d2a96f8fba4862",
  "morph_extended_max": "2d3fa2d75742014a425900a3e2d6ad7b2c66dfc50902b0bba5f6b18788db0b24",
  "morph_extended_min": "8692db5f583dfb81e27316766bb03c1007c2f2f71e7ff062c20ac62bd3a002c1",
  "gaussian_blur -> grayscale": "6c79f97d073a937cdb87f253b62bc138b9d88e3063133a04751cd46bb23c047f"
};

test("gray-order. frozen mirrored exports match bcdbf4a except approved rgb_adjust rounding", () => {
  for (const id of [...MIRRORED_OP_IDS, "gaussian_blur -> grayscale"]) {
    let pipe;
    if (id === "gaussian_blur -> grayscale") pipe = [step("gaussian_blur"), step("grayscale")];
    else {
      const st = step(id);
      if (["psnr", "ssim", "delta_e", "seg_score", "logical_ops", "mask", "inpaint", "hist_match", "template_matching", "ecc_register"].includes(id)) st.src2 = { pipeline: [] };
      pipe = [st];
    }
    const hash = crypto.createHash("sha256").update(buildPythonCode(pipe)).digest("hex");
    assert.equal(hash, MIRRORED_SHA256[id], id + ": mirrored export changed from its pinned baseline");
  }
  assert.ok(buildPythonCode([step("grayscale")]).includes("img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)"));
});

// DOM ขนาดเล็กแบบ ui-phase: เดินผ่าน createPalettePanel จริงทั้งรายการปกติและใช้ล่าสุด
class El {
  constructor(tag) { this.tagName = tag; this.children = []; this.parentNode = null; this.className = ""; this.style = { setProperty() {} }; this._listeners = {}; this._attrs = {}; this.hidden = false; }
  set textContent(v) { this._text = String(v); this.children = []; }
  get textContent() { return this.children.length ? this.children.map((c) => c.textContent).join("") : this._text || ""; }
  set innerHTML(v) { this.children = []; this._text = v; }
  get isConnected() { return !!this.parentNode; }
  appendChild(c) { c.parentNode = this; this.children.push(c); return c; }
  append(...cs) { for (const c of cs) this.appendChild(c); }
  replaceWith(n) { const p = this.parentNode; p.children[p.children.indexOf(this)] = n; n.parentNode = p; this.parentNode = null; }
  setAttribute(k, v) { this._attrs[k] = v; }
  addEventListener(k, fn) { (this._listeners[k] ||= []).push(fn); }
  fire(k, e = {}) { for (const fn of this._listeners[k] || []) fn(e); }
  focus() {} select() {}
  get classList() { return { add: (c) => { this.className += " " + c; }, remove: (c) => { this.className = this.className.split(" ").filter((x) => x !== c).join(" "); }, contains: (c) => this.className.split(" ").includes(c) }; }
  all(cls) { return this.children.flatMap((c) => [...c.className.split(" ").includes(cls) ? [c] : [], ...c.all(cls)]); }
  querySelector(s) { return this.all(s.slice(1))[0] || null; }
}
test("piqe-B. normal/recent palette tiles have one input; reference metrics still have two", () => {
  const prev = globalThis.document;
  globalThis.document = { createElement: (tag) => new El(tag) };
  try {
    const container = new El("div");
    const panel = createPalettePanel(container, { onAdd() {}, onAddDisplay() {}, onTemplates() {}, onSaveTemplate() {}, onAddDL() {} });
    panel.render();
    const search = container.all("palette-search")[0];
    // ค้นหาให้แสดงทุก op แบบ inline ไม่ต้องใช้ตำแหน่ง popup ใน DOM จำลอง
    for (const id of ["piqe", "psnr", "ssim", "delta_e", "seg_score"]) {
      search.value = id;
      search.fire("input");
      const row = container.all("tool-row").find((r) => r.title?.startsWith(OPS[id].name + " —"));
      assert.ok(row, id);
      assert.equal(row.all("af-port-in").length, id === "piqe" ? 1 : 2, id);
      row.fire("click");
      const recent = container.all("palette-recent")[0].all("tool-row").find((r) => r.title?.startsWith(OPS[id].name + " —"));
      assert.equal(recent.all("af-port-in").length, id === "piqe" ? 1 : 2, "recent " + id);
    }
    assert.equal(FILTERS.image_quality.inputs, 2);
  } finally { if (prev === undefined) delete globalThis.document; else globalThis.document = prev; }
});

test("piqe-B. editor dims connected/disconnected B, retains the slot, and restores PSNR styling", () => {
  const src = fs.readFileSync(new URL("../src/renderer/graph/editor.js", import.meta.url), "utf8");
  const body = src.slice(src.indexOf("function styleSlots("), src.indexOf("function installCardHooks("));
  const style = new Function("LiteGraph", "OPS", "currentOpId", "gc", body + "; return styleSlots;")({ CIRCLE_SHAPE: 3 }, OPS, currentOpId, () => ({ statusOff: "#999" }));
  const slot = { name: "เข้า B", link: 42 }, node = { filterId: "image_quality", properties: step("piqe").params, inputs: [{}, slot], outputs: [{}] };
  style(node);
  assert.equal(node.inputs.length, 2);
  assert.equal(slot.link, 42);
  assert.equal(slot.color_on, "#999");
  assert.equal(slot.color_off, "#999");
  assert.equal(slot.label, "ไม่ใช้ในวิธีนี้");
  assert.equal(slot.mvUnused, true);
  node.properties = step("psnr").params;
  style(node);
  assert.equal(slot.color_on, undefined);
  assert.equal(slot.color_off, undefined);
  assert.equal(slot.label, undefined);
  assert.equal(slot.mvUnused, undefined);
  assert.match(src, /onDrawForeground = function \(ctx\) \{\s*styleSlots\(this\)/);
});

test("piqe-B. hint follows the left B port in local coordinates without covering card text", () => {
  const src = fs.readFileSync(new URL("../src/renderer/graph/editor.js", import.meta.url), "utf8");
  const body = src.slice(src.indexOf("const NC_STACK"), src.indexOf("const SOURCE_KINDS"));
  const { styleSlots, drawCard, cardConnectionPos, LAYOUT_STD } = new Function(
    "LiteGraph", "OPS", "currentOpId", "graphColors", "shortToolName", "contrastOn", "ellipsisText",
    body + "; return { styleSlots, drawCard, cardConnectionPos, LAYOUT_STD };",
  )({ CIRCLE_SHAPE: 3 }, OPS, currentOpId, () => ({ statusOff: "#999" }), (s) => s, () => "#fff", (_ctx, _node, _key, text) => text);
  for (const link of [null, 42]) for (const shifted of [false, true]) {
    const layout = shifted ? { ...LAYOUT_STD, box: { ...LAYOUT_STD.box, y: 14, h: 56 } } : LAYOUT_STD;
    const slot = { name: "เข้า B", link };
    const node = { pos: shifted ? [-120, 65] : [340, 190], mvLayout: layout, title: "PIQE", filterId: "image_quality", properties: step("piqe").params, inputs: [{}, slot], outputs: [{}] };
    const texts = [], ctx = new Proxy({
      fillText(text, x, y) { texts.push({ text, x, y, align: this.textAlign, baseline: this.textBaseline }); },
    }, { get(target, key) { return key in target ? target[key] : () => {}; } });
    styleSlots(node);
    drawCard(ctx, node, { accent: "#aaa", glyph: "Q", sub: "score = 0" });
    const hint = texts.find((t) => t.text === "B · ไม่ใช้ในวิธีนี้");
    assert.ok(hint, "unused B hint is visible with link=" + link);
    const port = cardConnectionPos.call(node, true, 1);
    assert.equal(hint.align, "right", "hint stays to the left of the card");
    assert.equal(hint.x, port[0] - node.pos[0] - 8);
    assert.equal(hint.y, port[1] - node.pos[1]);
    assert.equal(hint.baseline, "middle");
    assert.ok(hint.x < 0 && hint.y - 6 > 0, "hint clears the title and card text");
    slot.label = "unused B"; // Display wording must not control whether the hint is drawn.
    texts.length = 0;
    drawCard(ctx, node, { accent: "#aaa", sub: "score = 0" });
    assert.ok(texts.some((t) => t.text === "B · unused B"));
    node.properties = step("psnr").params;
    styleSlots(node);
    texts.length = 0;
    drawCard(ctx, node, { accent: "#aaa", sub: "score = 0" });
    assert.ok(!texts.some((t) => t.text.startsWith("B · ")), "PSNR clears the unused hint");
    assert.equal(slot.link, link);
  }
});

test("piqe-B. worker skips B pixels/notes/state/memo, sweeps unused state, and resumes B for PSNR", () => {
  const src = fixture(), store = new Map(), memo = new Map();
  const b = { ...step("otsu_threshold"), nodeId: 81 };
  const pipe = two("piqe", [], [b]); pipe[0].nodeId = 80;
  let disposed = 0;
  store.set(81, { __dispose() { disposed++; } });
  try {
    const active = new Set(); collectNodeIds(pipe, active); sweepNodeState(active, store);
    assert.deepEqual([...active], [80]);
    assert.equal(disposed, 1);
    const expectedCtx = {}, expected = OPS.piqe.apply(cv, src, {}, expectedCtx);
    const first = runPipeline(src, pipe, store, active, memo);
    try {
      assert.deepEqual(Buffer.from(first.mat.data), Buffer.from(expected.data));
      assert.deepEqual(first.notes.find((n) => n.nodeId === 80 && n.values).values, expectedCtx.values);
      assert.ok(!first.notes.some((n) => n.nodeId === 81));
      assert.ok(!store.has(81)); assert.equal(memo.size, 0);
      const state = store.get(80);
      b.enabled = false;
      const again = runPipeline(src, pipe, store, active, memo);
      again.mat.delete();
      assert.equal(store.get(80), state, "unused B edits must not reset A state");
      b.enabled = true;
    } finally { first.mat.delete(); expected.delete(); }
    const psnr = two("psnr", [], [b]); psnr[0].nodeId = 80;
    const result = runPipeline(src, psnr, store, new Set([80, 81]), memo);
    const referenceB = runPipeline(src, [b], new Map(), new Set([81]));
    const gray = new cv.Mat(); cv.cvtColor(src, gray, cv.COLOR_RGB2GRAY);
    const ctx = { src2: referenceB.mat }, reference = FILTERS.image_quality.apply(cv, gray, psnr[0].params, ctx);
    try {
      assert.ok(store.has(81));
      assert.ok(result.notes.some((n) => n.nodeId === 81));
      assert.deepEqual(Buffer.from(result.mat.data), Buffer.from(reference.data));
      assert.deepEqual(result.notes.find((n) => n.nodeId === 80 && n.values).values, ctx.values);
      assertWrapperBalance(cv, "PIQE ignored B wrappers", () => {
        const out = runPipeline(src, pipe, new Map(), new Set([80])); out.mat.delete();
      }, { global: true });
    } finally { result.mat.delete(); referenceB.mat.delete(); reference.delete(); gray.delete(); }
    // สายเดียวกันที่ยังใช้อยู่ในผลลัพธ์อื่นต้องไม่ถูก sweep ทิ้ง
    const shared = new Set(); collectNodeIds(pipe, shared); collectNodeIds([b], shared);
    sweepNodeState(shared, store); assert.ok(store.has(81));
    const filter = FILTERS[b.id], apply = filter.apply;
    filter.apply = () => { throw new Error("B evaluated"); };
    try {
      const out = runPipeline(src, pipe, new Map(), new Set()); out.mat.delete();
      assert.throws(() => runPipeline(src, psnr, new Map(), new Set()), /B evaluated/);
    } finally { filter.apply = apply; }
  } finally { src.delete(); sweepNodeState(new Set(), store); }
});

test("piqe-B. stateful B is dormant for PIQE, memoized for PSNR, and disposed when unused", () => {
  const src = fixture();
  try {
    assertWrapperBalance(cv, "stateful B lifetime", () => {
      const store = new Map(), memo = new Map();
      const b = { ...step("mog2"), nodeId: 82 };
      const piqe = two("piqe", [], [b]); piqe[0].nodeId = 83;
      const psnr = two("psnr", [], [b]); psnr[0].nodeId = 83;
      try {
        const ignored = runPipeline(src, piqe, store, new Set([83]), memo); ignored.mat.delete();
        assert.ok(!store.has(82)); assert.ok(!memo.has(82));
        const first = runPipeline(src, psnr, store, new Set([82, 83]), memo);
        const cached = memo.get(82);
        assert.ok(cached && store.has(82));
        const second = runPipeline(src, psnr, store, new Set([82, 83]), memo);
        try {
          assert.equal(memo.get(82), cached, "shared B is evaluated once per frame");
          assert.deepEqual(second.notes, first.notes);
          assert.deepEqual(Buffer.from(second.mat.data), Buffer.from(first.mat.data));
        } finally { first.mat.delete(); second.mat.delete(); }
        const state = store.get(82), active = new Set();
        collectNodeIds(piqe, active); sweepNodeState(active, store);
        assert.ok(!store.has(82));
        assert.equal(state.mog, null, "MOG2 disposer clears its detector");
      } finally {
        for (const entry of memo.values()) entry.mat.delete();
        sweepNodeState(new Set(), store);
      }
    }, { global: true });
  } finally { src.delete(); }
});

let pyAvailable = false;
let syntaxPython = false;
try {
  execFileSync("python", ["-c", "pass"], { stdio: "pipe" }); syntaxPython = true;
  try { execFileSync("python", ["-c", "import cv2,numpy"], { stdio: "pipe" }); pyAvailable = true; } catch {}
} catch {}
const PY_SKIP = { skip: !pyAvailable && "python/cv2/numpy unavailable" };
test("gray-order-P-syntax. main/B/nested app-order exports compile without importing cv2", { skip: !syntaxPython && "python unavailable" }, () => {
  const pipes = ["hsv", "hls", "lab", "luv", "ycrcb", "xyz", "yuv"].map((space) => [color(space), step("grayscale")]);
  for (const id of ["otsu_threshold", "canny", "focus_score", "watershed"]) pipes.push([color(), step(id)]);
  for (const id of ["logical_ops", "inpaint", "mask", "hist_match", "template_matching", "psnr", "ssim", "seg_score", "ecc_register"]) pipes.push(two(id, [color()], [color("lab")]));
  pipes.push(two("psnr", [], two("logical_ops", [color()], [color("lab")])));
  execFileSync("python", ["-c", "import sys,json; [compile(c, 'gray_order', 'exec') for c in json.loads(sys.stdin.buffer.read())]"], { input: JSON.stringify(pipes.map((p) => buildPythonCode(p))), stdio: "pipe" });
});
function pythonCase(pipe, src, tail = "") {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gray-order-parity-"));
  const bgr = new cv.Mat();
  try {
    // cv2.imread ให้ BGR 3 ช่องเสมอ ใช้ข้อมูล 3 ช่องแม้ pipeline จะได้ภาพเทาภายหลัง
    cv.cvtColor(src, bgr, cv.COLOR_RGB2BGR);
    const input = path.join(dir, "input.bin"), output = path.join(dir, "output.bin");
    fs.writeFileSync(input, Buffer.from(bgr.data));
    let code = buildPythonCode(pipe);
    code = code.slice(0, code.indexOf('if __name__ == "__main__":'));
    for (const name of ["path", "path_a", "path_b"]) code = code.replace(`cv2.imread(${name})`, `np.fromfile(${name}, dtype=np.uint8).reshape((${src.rows}, ${src.cols}, 3))`);
    if (tail) code = code.replace("\n    return img\n", "\n    " + tail + "\n    return img\n");
    const args = code.includes("def process(path_a, path_b)") ? `${JSON.stringify(input)}, ${JSON.stringify(input)}` : JSON.stringify(input);
    code += `\nr = process(${args})\nr.tofile(${JSON.stringify(output)})\nprint('SHAPE', *r.shape)\n`;
    const file = path.join(dir, "case.py"); fs.writeFileSync(file, code);
    const stdout = execFileSync("python", [file], { encoding: "utf8", env: { ...process.env, PYTHONIOENCODING: "utf8" } });
    const bytes = fs.readFileSync(output), shape = stdout.match(/SHAPE (.*)/)[1].split(" ").map(Number);
    if (shape.length === 3) for (let i = 0; i < bytes.length; i += 3) [bytes[i], bytes[i + 2]] = [bytes[i + 2], bytes[i]];
    return { bytes, shape, stdout };
  } finally { bgr.delete(); fs.rmSync(dir, { recursive: true, force: true }); }
}
function parity(pipe, tail = "") {
  const src = fixture(); let out;
  try {
    out = runPipeline(src, pipe, new Map(), new Set());
    const py = pythonCase(pipe, src, tail);
    assert.deepEqual(py.shape, out.mat.channels() === 1 ? [out.mat.rows, out.mat.cols] : [out.mat.rows, out.mat.cols, out.mat.channels()]);
    assert.deepEqual(py.bytes, Buffer.from(out.mat.data), "0 bytes different");
    if (tail) {
      const fields = out.notes.at(-1).values.fields;
      assert.ok(fields.some((f) => Math.abs(f.value - Number(py.stdout.match(/VALUE (.*)/)[1])) < 1e-7));
    }
  } finally { if (out) out.mat.delete(); src.delete(); }
}
for (const space of ["hsv", "hls", "lab", "luv", "ycrcb", "xyz", "yuv"]) test("gray-order-P. " + space + " -> grayscale: 0 bytes different", PY_SKIP, () => parity([color(space), step("grayscale")]));
for (const id of ["otsu_threshold", "canny", "focus_score"]) test("gray-order-P. hsv -> " + id + ": 0 bytes different", PY_SKIP, () => parity([color(), step(id)], id === "focus_score" ? "print('VALUE', score)" : ""));
test("gray-order-P. hsv -> watershed -> grayscale: 0 bytes different", PY_SKIP, () => parity([color(), step("watershed"), step("grayscale")]));
for (const pipe of [[step("grayscale")], [step("gaussian_blur"), step("grayscale")]]) test("gray-order-P. mirrored " + pipe.map(opIdOfStep).join(" -> ") + ": 0 bytes different", PY_SKIP, () => parity(pipe));
for (const [name, pipe] of [
  ["A auto", two("psnr", [color(), step("canny")])],
  ["B explicit", two("psnr", [], [color(), step("grayscale")])],
  ["B auto", two("psnr", [], [color(), step("canny")])],
  ["independent A/B", two("logical_ops", [color()], [color("lab")])],
  ["nested B", two("psnr", [], two("logical_ops", [color()], [color("lab")]))],
]) test("gray-order-P. two-image " + name + ": 0 bytes different", PY_SKIP, () => parity(pipe));
