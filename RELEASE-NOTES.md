• เพิ่มเครื่องมือแก้ภาพเบลอ: Wiener และ Lucy-Richardson Deconvolution (เลือกรูปแบบการเบลอได้ เช่น เบลอจากการเคลื่อนที่)

• เพิ่มเครื่องมือวิเคราะห์ลวดลายพื้นผิว: Gabor Filter และ GLCM (วัดค่า contrast, energy, homogeneity, correlation)

• เพิ่ม Image Registration (ECC) จัดภาพสองภาพให้ทับกันตรงก่อนนำไปเทียบหาจุดต่าง

• เพิ่มตัวลด noise แบบ Non-Local Means, Dehaze (ลดหมอก), และ DCT / จำลองการบีบอัดแบบ JPEG

• Morphology มีโหมดใหม่: Hit-or-Miss, Regional Max / Min, H-Maxima และ Extended Max

• เพิ่ม Quantize (ลดจำนวนระดับสี), กรองวัตถุตามรูปร่าง, และค่าวัดใหม่ใน Region Properties (Solidity, Eccentricity, แกนยาว/สั้น, Extent, เส้นผ่านศูนย์กลางเทียบเท่า, ความสว่างเฉลี่ย)

• Resize / Rotate เลือกวิธีประมาณค่าพิกเซลได้ (Nearest / Bilinear / Bicubic)

• เพิ่มตัวเลือกในเครื่องมือเดิมอีกหลายอย่าง: Guided Filter, Speckle Noise, Log Transform, Triangle Threshold, Butterworth Filter, Pseudocolor, White Balance แบบอื่น, วัดความต่างสี ΔE, ปริภูมิสี XYZ / Luv / YUV, วัดความคมชัด (Focus), ใช้ฟิลเตอร์เฉพาะใน ROI และแปลงพิกเซลเป็นมิลลิเมตร

• แก้ชื่อคลาสภาษาไทยจากโมเดล ONNX ให้แสดงถูกต้อง

• โค้ดเทรน RF-DETR เทรนจากไฟล์ dataset แบบ YOLO ที่ export จากแอปได้โดยตรง
